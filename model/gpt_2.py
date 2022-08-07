import tarfile
import os
import json
import requests
import sys
import shutil
import re
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
import time
from datetime import datetime
import csv
import argparse

import gpt_2_simple

from patches.sample import sample_sequence
gpt_2_simple.src.sample.sample_sequence = sample_sequence

from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer

def predict(sess,
            text,
            run_name='run1',
            checkpoint_dir='checkpoint',
            model_name=None,
            model_dir='models',
            sample_dir='samples',
            seed=None,
            temperature=1,
            top_k=2):

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()

    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])
    context_tokens = enc.encode(text)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    start_time = time.time()

    proba_t = sample.sample_sequence(
        hparams=hparams,
        context=context,
        temperature=temperature
    )

    start_time = time.time()

    proba = sess.run(proba_t, feed_dict={
            context: [context_tokens]
        })

    top_k_idxs = np.flip(np.argsort(proba))[:top_k]

    top_k_tokens = enc.decode(top_k_idxs).split()
    top_k_proba = proba[top_k_idxs]

    return top_k_tokens, top_k_proba
