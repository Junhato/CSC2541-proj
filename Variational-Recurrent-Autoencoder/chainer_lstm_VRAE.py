import os
import time
import numpy as np


from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F


class VRAE(FunctionSet):

    def __init__(self, n_input, n_hidden, n_latent, loss_func):
        """
        :param n_input: number of input dimensions
        :param n_hidden: number of LSTM cells for both generator and decoder
        :param n_latent: number of dimensions for latent code (z)
        :param loss_func: loss function to compute reconstruction error (e.g. F.mean_squared_error)
        """
        self.__dict__.update(locals())
        super(VRAE, self).__init__(

            # Encoder (recognition):
            recog_x_h=F.Linear(n_input, n_hidden*4),
            recog_h_h=F.Linear(n_hidden, n_hidden*4),
            recog_mean=F.Linear(n_hidden, n_latent),
            recog_log_sigma=F.Linear(n_hidden, n_latent),

            # Decoder (generation)
            gen_z_h=F.Linear(n_latent, n_hidden*4),
            gen_x_h=F.Linear(n_input, n_hidden*4),
            gen_h_h=F.Linear(n_hidden, n_hidden*4),
            output=F.Linear(n_hidden, n_input)
        )

    def softplus(self, x):
        return F.log(F.exp(x) + 1)

    def binary_cross_entropy(self, y, t):
        ya = y.data
        ta = t.data
        d  = -1*np.sum(ya*np.log(ta) + (1-ya)*np.log(1-ta))
        v_d = Variable(np.array(d).astype(np.float32))
        return v_d

    def forward_one_step(self, x_data, state):
        """
        Does encode/decode on x_data.
        :param x_data: input data (a single timestep) as a numpy.ndarray
        :param state: previous state of RNN
        :param nonlinear_q: nonlinearity used in q(z|x) (encoder)
        :param nonlinear_p: nonlinearity used in p(x|z) (decoder)
        :param output_f: #TODO#
        :return: output, recognition loss, KL Divergence, state
        """
        #=====[ Step 1: Compute q(z|x) - encoding step, get z ]=====
        # Forward encoding
        for i in range(x_data.shape[0]):
            x = F.dropout(Variable(x_data[i].reshape((1, x_data.shape[1]))), 0.5)
            h_in = F.dropout(self.recog_x_h(x) + self.recog_h_h(state['h_rec']), 0.5)
            c_t, h_t = F.lstm(state['c_rec'], h_in)
            state.update({'c_rec':c_t, 'h_rec':h_t})
        # Compute q_mean and q_log_sigma
        q_mean = self.recog_mean( state['h_rec'] )
        q_log_sigma = 0.5 * self.recog_log_sigma( state['h_rec'] )
        # Compute KL divergence based on q_mean and q_log_sigma
        KLD = -0.0005 * F.sum(1 + q_log_sigma - q_mean**2 - F.exp(q_log_sigma))
        # Compute as q_mean + noise*exp(q_log_sigma)
        eps = Variable(np.random.normal(0, 1, q_log_sigma.data.shape ).astype(np.float32))
        z   = q_mean + F.exp(q_log_sigma) * eps

        #=====[ Step 2: Compute p(x|z) - decoding step ]=====
        # Initial step
        output = []
        h_in = self.gen_z_h(z)
        c_t, h_t = F.lstm(state['c_gen'], h_in)
        state.update({'c_gen':c_t, 'h_gen':h_t})
        rec_loss = Variable(np.zeros((), dtype=np.float32))
        for i in range(x_data.shape[0]):
            # Get output and loss
            x_t = self.output(h_t)
            output.append(x_t.data)
            rec_loss += self.loss_func(x_t, Variable(x_data[i].reshape((1, x_data.shape[1])).astype(np.int32)))
            # Get next hidden state
            h_in = self.gen_x_h(x_t) + self.gen_h_h(state['h_gen'])
            c_t, h_t = F.lstm(state['c_gen'], h_in)
            h_t = F.softmax(h_t)
            state.update({'c_gen':c_t, 'h_gen':h_t})

        #=====[ Step 3: Compute KL-Divergence based on all terms ]=====
        return output, rec_loss, KLD, state

    def generate_z_x(self, seq_length_per_z, sample_z, nonlinear_q='tanh', nonlinear_p='tanh', output_f='sigmoid', gpu=-1):

        output = np.zeros((seq_length_per_z * sample_z.shape[0], self.recog_in_h.W.shape[1]))

        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_a_f = nonlinear[output_f]

        for epoch in xrange(sample_z.shape[0]):
            gen_out = np.zeros((seq_length_per_z, output.shape[1]))
            z = Variable(sample_z[epoch].reshape((1, sample_z.shape[1])))

            # compute p( x | z)
            h0 = nonlinear_f_p( self.z(z) )
            x_gen_0 = output_a_f( self.output(h0) )
            state['gen_h'] = h0
            if gpu >= 0:
                np_x_gen_0 = cuda.to_cpu(x_gen_0.data)
                gen_out[0] = np_x_gen_0
            else:
                gen_out[0] = x_gen_0.data

            x_t_1 = x_gen_0

            for i in range(1, seq_length_per_z):
                hidden_p_t = nonlinear_f_p( self.gen_in_h(x_t_1) + self.gen_h_h(state['gen_h']) )
                output_t   = output_a_f( self.output(hidden_p_t) )
                if gpu >= 0:
                    np_output_t = cuda.to_cpu(output_t.data)
                    gen_out[i] = np_output_t
                else:
                    gen_out[i]  = output_t.data
                state['gen_h'] = hidden_p_t
                x_t_1 = output_t

            output[epoch*seq_length_per_z+1:(epoch+1)*seq_length_per_z, :] = gen_out[1:]

        return output

    def make_initial_state(self):
        """Returns an initial state of the RNN - all zeros"""
        return {
            'h_rec': Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'c_rec': Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'h_gen': Variable(np.zeros((1, self.n_hidden), dtype=np.float32)),
            'c_gen': Variable(np.zeros((1, self.n_hidden), dtype=np.float32))
        }
