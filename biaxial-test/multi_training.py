import os, random
from midi_to_statematrix import *
from data import *
import cPickle as pickle

import signal

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations

def loadPieces(dirpath):

    pieces = {}

    for fname in os.listdir(dirpath):
        name, ext = os.path.splitext(fname)
        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < batch_len:
            continue

        pieces[name] = outMatrix
        print "Loaded {}".format(name)

    return pieces

def getPieceSegment(pieces):
    piece_output = random.choice(pieces.values())
    start = random.randrange(0,len(piece_output)-batch_len,division_len)
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start)

    seg_out = piece_output[start:start+batch_len]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def getPieceBatch(pieces):
    i,o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return numpy.array(i), numpy.array(o)

def trainPiece(model,pieces,epochs,start=0):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        asdf_in, asdf_out = getPieceBatch(pieces)
        print(asdf_in.shape)
        print(asdf_out.shape)
        input_slice = asdf_in[:,0:-1]
        print(input_slice.shape)
        n_batch, n_time, n_note, n_ipn = input_slice.shape
        time_inputs = input_slice.transpose((1,0,2,3)).reshape((n_time,n_batch*n_note,n_ipn))
        print(time_inputs.shape)
        num_time_parallel = time_inputs.shape[1]
        
        #error = model.update_fun(*getPieceBatch(pieces))
        #if i % 100 == 0:
        #    print "epoch {}, error={}".format(i,error)
        #if i % 500 == 0 or (i % 100 == 0 and i < 1000):
        #    xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
        #    noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),'output/sample{}'.format(i))
        #    pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    #signal.signal(signal.SIGINT, old_handler)