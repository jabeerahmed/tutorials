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

def main():
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(model_fn=mnist.model_fn, 
	                                          model_dir="/tmp/mnist_convnet_model")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	mnist_classifier.train(input_fn=mnist.input_fn(TRAIN, num_epochs=10, shuffle=True),
	                       hooks=[logging_hook])

	eval_results = mnist_classifier.evaluate(input_fn=mnist.input_fn(EVAL))

# Our application logic will be added here
if __name__ == "__main__":
    tf.app.run()
