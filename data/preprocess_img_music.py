from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
from os import listdir
from midi_manipulation import *
import json
import math
import os.path
import random
import sys
import threading

import numpy as np
import tensorflow as tf

class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _midi_chunks_matrix(filename):
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

def _get_image_music_pair(image, decoder, chunk):
  """Return [image,chunk] pairing."""

  with tf.gfile.FastGFile(image, "r") as f:
    encoded_image = f.read()

  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return
  
  return [image,chunk]
  
def get_image_music_pairs():
  mood_dirs = ['sad', 'anxious', 'happy'] 
  decoder = ImageDecoder()
  images_and_music = []
    
  for mood_dir in mood_dirs:
    images = os.listdir(mood_dir + '/images/')
    midis = [f for f in os.listdir(mood_dir + '/music/') if os.path.isfile(mood_dir+'/music/'+f)]
        
    for image in images:
      if ".jpg" in image:
        impath = mood_dir + '/images/' + image
        mupath = random.choice(midis)
        midi_chunks = _midi_chunks_matrix(mood_dir + '/music/' + mupath)
        for chunk in midi_chunks:
          if len(chunk) > 0:
            images_and_music.append(_get_image_music_pair(impath, decoder, chunk))

  return images_and_music

