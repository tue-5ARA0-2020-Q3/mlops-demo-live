# import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback
#import dvclive

from . import utils

"""
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_step()
"""

class Learner:
    """ Class to train & save the model """
    def __init__(self, output_dir):
        self.folder_ml = output_dir
        self.model = None
        utils.enable_gpu_memory_growth()

    def make_model(self):
        """
        Creates a model
        """
        self.model = Sequential(
            [
                Input(shape=(150, 150, 3)),

                Conv2D(32, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.5),

                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid'),
            ])

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train_model(self, hyper_param=dict()):
        """
        Training the model
        """
        batch_size = hyper_param.get('batch_size', 32)
        epochs = hyper_param.get('epochs', 1)

        self.make_model()

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')

        print('Fitting model...')
        #dvclive.init("training_metrics")
        self.model.fit(train_generator,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=validation_generator,
                       verbose=2)
                       #callbacks=[MetricsCallback()])

    def save_model(self, model_name):
        """
        Saving the model using pickle
        :param model_name: name of model
        :type model_name: str
        """
        filename = model_name + '.h5'
        self.model.save_weights(self.folder_ml + filename)
