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
from chainer_VRAE import VRAE

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
midi = dataset.load_midi_data('%s/temp/music/c-major-scale.mid' % args.data_path)
train_x = midi[:120].astype(np.float32)

n_input = train_x.shape[1]
n_hidden = 125
n_latent = 2

frames = train_x.shape[0]
n_batch = 6
seq_length = frames / n_batch

split_x = np.vsplit(train_x, n_batch)

n_epochs = 2000
continuous = False

loss_func = F.sigmoid_cross_entropy


if args.init_from == "":
    model = VRAE(n_input, n_hidden, n_latent, loss_func)
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
    outputs = np.zeros(train_x.shape, dtype=np.float32)
    # state = make_initial_state(n_hidden_recog[0], state_pattern)
    for i in xrange(n_batch):
        state = model.make_initial_state()
        x_batch = split_x[i]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)

        output, rec_loss, kl_loss, state = model.forward_one_step(x_batch, state)

        outputs[i * seq_length:(i + 1) * seq_length, :] = output

        loss = rec_loss + kl_loss
        total_loss += loss
        total_rec_loss += rec_loss
        total_losses[epoch-1] = total_loss.data
        
        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.clip_grads(args.clip_grads)
        optimizer.update()

    saved_output = outputs

    print "{}/{}, train_loss = {}, total_rec_loss = {}, time = {}".format(epoch, n_epochs, total_loss.data,
                                                                              total_rec_loss.data, time.time() - t1)

    if epoch % 100 == 0:
        dataset.write_to_file(np.round(saved_output), epoch)
        model_path = "%s/VRAE_%s_%d.pkl" % (args.output_dir, args.dataset, epoch)
        with open(model_path, "w") as f:
            pickle.dump(copy.deepcopy(model).to_cpu(), f)
