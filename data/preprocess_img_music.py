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


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _to_sequence_example(image, counter, decoder, music):
  """Builds a SequenceExample proto for an image-caption pair.

  Args:
    image: Image filename.
    decoder: An ImageDecoder object.
    music: A chunk of music matrix. Dimensions are x * 156, where x <= 128 

  Returns:
    A SequenceExample proto.
  """
  with tf.gfile.FastGFile(image, "r") as f:
    encoded_image = f.read()

  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  context = tf.train.Features(feature={
    "image/image_id": _int64_feature(counter),
    "image/data": _bytes_feature(encoded_image),
  })
  
  # MUST PROCESS MUSIC HERE AS A FEATURE LIST BUT I HAVE NO IDEA HOW

  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example

if __name__ == "__main__":
  mood_dirs = ['sad', 'anxious', 'happy']
  counter = 0
  decoder = ImageDecoder()
  output_filename = "tfrecords"
  writer = tf.python_io.TFRecordWriter(output_filename)
    
  for mood_dir in mood_dirs:
    image_dir = os.listdir(mood_dir + '/images/')
    music_dir = os.listdir(mood_dir + '/music/')
        
    for image in image_dir:
      if ".jpg" in image:
        impath = mood_dir + '/images/' + image
        mupath = random.choice(music_dir)
        midi_chunks = midi_chunks_matrix(mood_dir + '/music/' + mupath)
        for chunk in midi_chunks:
          seq_ex = _to_sequence_example(impath, counter, decoder, chunk)
          if seq_ex is not None:
            writer.write(seq_ex.SerializeToString()) 
             
        counter += 1
  writer.close()
