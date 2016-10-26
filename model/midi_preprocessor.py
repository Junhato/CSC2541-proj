from midi_manipulation import *
import os, math

def midi_chunks_matrix(filename):
    """Cuts midi up into several chunks and returns an array of each chunk's note matrix.
    Note matrix is computed by midiToNoteStateMatrix from midi_manipulation"""
    
    note_matrix = midiToNoteStateMatrix(filename)
    matrices = []
    batch_len = 16*8 # length of each chunk
    num_chunks = int(math.ceil(len(note_matrix) / float(batch_len)))
    for i in range(num_chunks):
        end_index = -1
        if i * batch_len + batch_len < len(note_matrix):
            end_index = i * batch_len + batch_len
        matrices.append(note_matrix[i * batch_len : end_index])
        
    return matrices
 
if __name__ == "__main__":
    fname = "../data/unlabeled/music/08heatman.mid"
    print midi_chunks_matrix(fname)
