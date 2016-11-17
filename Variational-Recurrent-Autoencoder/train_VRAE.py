#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import six
import random

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

##### DATA #####
print("PROCESSING DATA")
mood_dirs = ["sad", "happy", "anxious"]
midi_list = []
for mood in mood_dirs:
    music_dir = os.listdir(args.data_path + mood + "/music")
    for music in music_dir:
        music_path = args.data_path + mood + "/music/" + music
        if os.path.isfile(music_path):
            midi = dataset.load_midi_data(music_path)
            for twenty_chunk in np.array_split(midi, 10):
                if len(twenty_chunk) == 10:
                    midi_list.append(twenty_chunk.astype(np.float32))
##### DATA #####

##### MODEL #####
print("PROCESSING MODEL")
n_epochs = 500
continuous = False
n_hidden = [500]
n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)
n_z = 2

layers = {}

# Recognition model.
example_midi = midi_list[0]
rec_layer_sizes = [(example_midi.shape[1], n_hidden_recog[0])]
rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])
rec_layer_sizes += [(n_hidden_recog[-1], n_z)]

layers['recog_in_h'] = F.Linear(example_midi.shape[1], n_hidden_recog[0], nobias=True)
layers['recog_h_h']  = F.Linear(n_hidden_recog[0], n_hidden_recog[0])

layers['recog_mean'] = F.Linear(n_hidden_recog[-1], n_z)
layers['recog_log_sigma'] = F.Linear(n_hidden_recog[-1], n_z)

# Generating model.
gen_layer_sizes = [(n_z, n_hidden_gen[0])]
gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])
gen_layer_sizes += [(n_hidden_gen[-1], example_midi.shape[1])]

layers['z'] = F.Linear(n_z, n_hidden_gen[0])
layers['gen_in_h'] = F.Linear(example_midi.shape[1], n_hidden_gen[0], nobias=True)
layers['gen_h_h']  = F.Linear(n_hidden_gen[0], n_hidden_gen[0])

layers['output']   = F.Linear(n_hidden_gen[-1], example_midi.shape[1])

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

# sample
outputs = np.zeros((120, example_midi.shape[1]), dtype=np.float32)
counter = 0
counter2 = 0

for epoch in range(0, n_epochs):
    total_loss = 0.0
    for i in range(len(midi_list)):
        t1 = time.time()
        state = make_initial_state(n_hidden_recog[0], state_pattern)
        x_batch = midi_list[i]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)

        output, rec_loss, kl_loss, state = model.forward_one_step(x_batch, state, continuous, nonlinear_q='tanh', nonlinear_p='tanh', output_f = 'sigmoid', gpu=-1)

        loss = rec_loss + kl_loss
        total_loss += loss
        
        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.clip_grads(args.clip_grads)
        optimizer.update()
        
        outputs[counter * 10 : (counter + 1) * 10] = np.round(output)
        counter += 1
        
        if counter == 12:
            dataset.write_to_file(outputs, counter2)
            print "{}, total_loss = {}, time = {}".format(counter2, total_loss.data, time.time()-t1)
            model_path = "%s/VRAE_%s_%d.pkl" % (args.output_dir, args.dataset, counter2)
            with open(model_path, "w") as f:
                pickle.dump(copy.deepcopy(model).to_cpu(), f)
            # reset
            counter = 0
            total_loss = 0
            counter2 += 1
