"""Training script for the WaveNet network."""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

#import tensorflow as tf
import sugartensor as tf

from wavenet import WaveNetModel, TextReader

BATCH_SIZE = 1
DATA_DIRECTORY = './data'
CHECKPOINT_EVERY = 500
NUM_STEPS = 4000
LEARNING_RATE = 0.001
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 1000
L2_REGULARIZATION_STRENGTH = 0


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut text samples to this many '
                        'samples.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Disabled by default')
    return parser.parse_args()


def main():
    args = get_arguments()

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    tf.sg_verbosity(10)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw text.
    with tf.name_scope('create_inputs'):
        reader = TextReader(
            args.data_dir,
            coord,
            sample_size=args.sample_size)
        text_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"])
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    loss = net.loss(text_batch, args.l2_regularization_strength)
    
    old_set = tf.sg_set_train    
    def my_set(sess):        
        old_set(sess)
        reader.maybe_start_threads(sess)        
    tf.sg_set_train = my_set

    try:
        tf.sg_train(ep_size=1000,log_interval=30,lr=0.01,loss=loss,console_log=True,save_dir="sg2")        

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()


if __name__ == '__main__':
    main()
