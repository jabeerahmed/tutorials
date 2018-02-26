import numpy as np
import tensorflow as tf


INPUT_BATCH_SIZE = 32
INPUT_NUM_EPOCHS = 1
INPUT_NUM_THREADS = 1
INPUT_SHUFFLE_DATA = False

PREDICT = tf.estimator.ModeKeys.PREDICT
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL

def input_fn(mode, batch_size=INPUT_BATCH_SIZE, 
                   num_epochs=INPUT_NUM_EPOCHS,
                   shuffle=INPUT_SHUFFLE_DATA,
                   num_threads=INPUT_NUM_THREADS):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    if mode == TRAIN:
        data = mnist.train.images
        labels = mnist.train.labels
    elif mode == EVAL:
        data = mnist.validation.images
        labels = mnist.validation.labels
    elif mode == PREDICT:
        data = mnist.test.images
        labels = mnist.test.labels

    labels = np.asarray(labels, dtype=np.int32)
    return tf.estimator.inputs.numpy_input_fn(
                x={'x': data}, 
                y=labels, 
                batch_size=batch_size, 
                num_epochs=num_epochs, 
                shuffle=shuffle, 
                num_threads=num_threads)


def model_fn(features, labels, mode, params, config):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, 
                                training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { 
        "accuracy": tf.metrics.accuracy(labels=labels, 
                                        predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      loss=loss, 
                                      eval_metric_ops=eval_metric_ops)


class MnistClassifier(tf.estimator.Estimator):

    def __init__(self, model_dir=None, config=None, warm_start_from=None):
        super(MnistClassifier, self).__init__(model_fn=model_fn, 
                                              model_dir=model_dir, 
                                              config=config)    
