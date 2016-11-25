import numpy as np
from chainer import Variable, Chain
from chainer import functions as F
from chainer import links as L

class MyLSTM(Chain):
    def __init__(self, n_input, n_hidden, n_latent):
        super(MyLSTM, self).__init__(
            encoder = L.LSTM(n_input, n_hidden),
            encoder_mean = F.Linear(n_hidden, n_latent),
            encoder_log_sigma = F.Linear(n_hidden, n_latent),
            latent_to_hidden = F.Linear(n_latent, n_hidden),
            decoded_to_hidden = F.Linear(n_input, n_hidden),
            decoder = L.LSTM(n_hidden, n_input),
        )
        
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        
    def forward(self, x_data):
        # Forward encoding
        for i in range(x_data.shape[0]):
            x = Variable(x_data[i].reshape((1, x_data.shape[1])))
            encoded_x = self.encoder(x)
            
        # Compute q_mean and q_log_sigma
        q_mean = self.encoder_mean(self.encoder.c)
        q_log_sigma = 0.5 * self.encoder_log_sigma(self.encoder.c)
        
        # Compute KL Divergence based on q_mean & q_log_sigma
        KLD = -0.1 * F.sum(1 + q_log_sigma - q_mean**2 - F.exp(q_log_sigma))
        
        # Compute as q_mean + noise * exp(q_log_sigma)
        eps = Variable(np.random.normal(0, 1, q_log_sigma.data.shape ).astype(np.float32))
        z   = q_mean + F.exp(q_log_sigma) * eps
        print(z.shape)
        
        # Decode
        output = []
        c_in = self.latent_to_hidden(z)
        rec_loss = Variable(np.zeros((), dtype=np.float32))
        print(c_in.shape)
        for i in range(x_data.shape[0]):
            decoded_x = self.decoder(c_in)
            output.append(decoded_x.data[0])
            c_in = self.decoded_to_hidden(self.decoder.c + decoded_x)
            rec_loss += F.sigmoid_cross_entropy(decoded_x, Variable(x_data[i].reshape((1, x_data.shape[1])).astype(np.int32)))
            
        return output, rec_loss, KLD
        
