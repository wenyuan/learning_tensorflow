#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

from text_classification import data_helpers
from text_classification.text_cnn import TextCNN

# params

# Data loading params
tf.flags.DEFINE_float('dev_sample_pct', 1, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_string('positive_data_file', './data/rt-polaritydata/rt-polarity.pos', 'Data source for the positive')
tf.flags.DEFINE_string('negative_data_file', './data/rt-polaritydata/rt-polarity.neg', 'Data source for the negative')

# Model Hparams
tf.flags.DEFINE_integer('embedding', 128, 'Dimension of character embedding')
tf.flags.DEFINE_string('filter_size', '3,4,5', 'Filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters')
tf.flags.DEFINE_float('dropout', 0.5, 'Dropout')
tf.flags.DEFINE_float('L2_reg_lambda', 0.0, 'L2')

# Training params
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size')
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate when every 100 epochs')
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save when every 100 epochs')

tf.flags.DEFINE_boolean('log_device_placement', True, 'if print log')
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow automatic distribution equipment(CPU/GPU...)')

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{attr}={attr}'.format(attr=attr.upper(), value=value))

# Load data
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

max_document_length = max(len(x.split(' ')) for x in x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))  # add char(default:0) to ensure each sample the same length

# Shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_pct * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
print('max_document_length: {0}'.format(len(vocab_processor.vocabulary_)))
print('train/dev split: {train}/{dev}'.format(train=len(y_train), dev=len(y_dev)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            embedding_size = FLAGS.embedding,
            filter_sizes = list(map(int, FLAGS.filter_size.split(','))),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, './', global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

if __name__ == "__main__":
    print('hello world')
