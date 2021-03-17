import tensorflow as tf


def enable_gpu_memory_growth():
    """
    Enable TensorFlow to use/allocate GPU memory
    :return: bool
    """
    enabled = False

    gpus_physical = tf.config.list_physical_devices('GPU')
    if gpus_physical:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus_physical:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpus_logical = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus_physical), "Physical GPUs,", len(gpus_logical), "Logical GPUs")
            enabled = True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    return enabled
