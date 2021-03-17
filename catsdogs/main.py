""" Module to run src files """
import tensorflow as tf
from . import train


def run():

    """
    Function to run the machine learning model
    """

    # evaluate model with single split
    learner = train.Learner()
    learner.train_model()
    learner.save_model("model_simple")


def enable_gpu_memory_growth(gpus_physical):
    if gpus_physical:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus_physical:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpus_logical = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus_physical), "Physical GPUs,", len(gpus_logical), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    enable_gpu_memory_growth(gpus)
    run()