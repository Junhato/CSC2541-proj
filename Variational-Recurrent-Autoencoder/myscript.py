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
from MyLSTM import MyLSTM

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
chunk_length = 20
overlap = 5
for mood in mood_dirs:
    music_dir = os.listdir(args.data_path + mood + "/music")
    for music in music_dir:
        music_path = args.data_path + mood + "/music/" + music
        if os.path.isfile(music_path):
            midi = dataset.load_midi_data(music_path)
            start = 0
            #while start + chunk_length <= len(midi):
            midi_list.append(midi[start: start+chunk_length].astype(np.float32))
            #start += overlap
##### DATA #####

##### MODEL #####
print("PROCESSING MODEL")
n_epochs = 500
continuous = False
n_hidden = [500]
n_z = 2

example_midi = midi_list[0]
if args.init_from == "":
    model = MyLSTM(example_midi.shape[1], n_hidden[0], n_z)
    model.reset_state()
else:
    model = pickle.load(open(args.init_from))

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# use Adam
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(0, n_epochs):
    for i in range(len(midi_list)):
        t1 = time.time()
        x_batch = midi_list[i]

        output, rec_loss, kl_loss = model.forward(x_batch)
        model.cleargrads()
        loss = rec_loss + kl_loss       
        loss.backward()
        optimizer.update()

        print "{}/{}, train_loss = {}, time = {}".format(i, len(midi_list), loss.data, time.time()-t1)

        if i % 10 == 0:
            dataset.write_to_file(np.round(output), epoch, i)
    
    model_path = "%s/VRAE_%s_%d.pkl" % (args.output_dir, args.dataset, epoch)
    with open(model_path, "w") as f:
        pickle.dump(copy.deepcopy(model).to_cpu(), f)
