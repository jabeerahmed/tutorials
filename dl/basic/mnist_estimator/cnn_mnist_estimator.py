from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import model as mnist

PREDICT = tf.estimator.ModeKeys.PREDICT
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_args):
    # Create the Estimator
    # mnist_classifier = tf.estimator.Estimator(model_fn=mnist.model_fn, 
    #                                           model_dir="model_dir")
    mnist_classifier = mnist.MnistClassifier(model_dir="model_dir")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
                                              every_n_iter=50)

    # Profiler
    builder = tf.profiler.ProfileOptionBuilder
    prof_dir = 'train_dir/'
    opts = builder(builder.time_and_memory()).order_by('micros').with_file_output(prof_dir+'dump').build()
    with tf.contrib.tfprof.ProfileContext(prof_dir, trace_steps=[500], dump_steps=[500], debug=True) as pctx:    # Train
        pctx.add_auto_profiling("graph", options=opts, profile_steps=[500])
        pctx.add_auto_profiling("op", options=opts, profile_steps=[500])
    # Profiler - End

        mnist_classifier.train(input_fn=mnist.input_fn(TRAIN, num_epochs=10, shuffle=True), 
                               steps=400,
                               hooks=[logging_hook])

    eval_results = mnist_classifier.evaluate(input_fn=mnist.input_fn(EVAL))

# Our application logic will be added here
if __name__ == "__main__":
    tf.app.run()


    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())

    #     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
