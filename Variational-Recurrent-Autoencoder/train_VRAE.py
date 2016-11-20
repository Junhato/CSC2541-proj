#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import six

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer_VRAE import VRAE, make_initial_state

import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default="../data/")
parser.add_argument('--output_dir',     type=str,   default="output")
parser.add_argument('--dataset',        type=str,   default="midi")
parser.add_argument('--init_from',      type=str,   default="")
parser.add_argument('--clip_grads',     type=int,   default=5)
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

### DATA ###
print("PROCESSING DATA")
mood_dirs = ['sad', 'happy', 'anxious']
midi_list = []
chunk_length = 20
overlap = 5
for mood in mood_dirs:
    music_dir = os.listdir(args.data_path + mood + "/music")
    for music in music_dir:
        music_path = args.data_path + mood + "/music/" + music
        if os.path.isfile(music_path):
            midi = dataset.load_midi_data(music_path)
            start = 0
            while start + chunk_length <= len(midi):
                midi_list.append(midi[start: start+chunk_length].astype(np.float32))
                start += overlap
print("FINISHED PROCESSING DATA")

### Recognition model ###
n_epochs = 500
n_hidden = [500]
n_z = 2
continuous = False

n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)

layers = {}
train_x = midi_list[0] #example midi
rec_layer_sizes = [(train_x.shape[1], n_hidden_recog[0])]
rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])
rec_layer_sizes += [(n_hidden_recog[-1], n_z)]

layers['recog_in_h'] = F.Linear(train_x.shape[1], n_hidden_recog[0], nobias=True)
layers['recog_h_h']  = F.Linear(n_hidden_recog[0], n_hidden_recog[0])

layers['recog_mean'] = F.Linear(n_hidden_recog[-1], n_z)
layers['recog_log_sigma'] = F.Linear(n_hidden_recog[-1], n_z)

# Generating model.
gen_layer_sizes = [(n_z, n_hidden_gen[0])]
gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])
gen_layer_sizes += [(n_hidden_gen[-1], train_x.shape[1])]

layers['z'] = F.Linear(n_z, n_hidden_gen[0])
layers['gen_in_h'] = F.Linear(train_x.shape[1], n_hidden_gen[0], nobias=True)
layers['gen_h_h']  = F.Linear(n_hidden_gen[0], n_hidden_gen[0])

layers['output']   = F.Linear(n_hidden_gen[-1], train_x.shape[1])

if args.init_from == "":
    model = VRAE(**layers)
else:
    model = pickle.load(open(args.init_from))

# state pattern
state_pattern = ['recog_h', 'gen_h']

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()


# use Adam
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

total_losses = np.zeros(n_epochs, dtype=np.float32)

for epoch in xrange(1, n_epochs + 1):
    print('epoch', epoch)

    t1 = time.time()
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    # state = make_initial_state(n_hidden_recog[0], state_pattern)
    for i in xrange(1):
        state = make_initial_state(n_hidden_recog[0], state_pattern)
        x_batch = midi_list[i]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)

        output, rec_loss, kl_loss, state = model.forward_one_step(x_batch, state, continuous, nonlinear_q='tanh', nonlinear_p='tanh', output_f = 'sigmoid', gpu=-1)

        loss = rec_loss + kl_loss
        total_loss += loss
        total_rec_loss += rec_loss
        total_losses[epoch-1] = total_loss.data

        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.clip_grads(args.clip_grads)
        optimizer.update()

   
        print "{}/{}, train_loss = {}, total_rec_loss = {}, time = {}".format(i, len(midi_list), total_loss.data, total_rec_loss.data, time.time()-t1)

        if i % 100 == 0:
            dataset.write_to_file(np.round(output), epoch, i)
            model_path = "%s/VRAE_%s_%d_%d.pkl" % (args.output_dir, args.dataset, epoch, i)
            with open(model_path, "w") as f:
                pickle.dump(copy.deepcopy(model).to_cpu(), f)
