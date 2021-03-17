from . import train


def run_ml_training():
    """
    Function to run the machine learning model
    """
    # evaluate model with single split
    learner = train.Learner()
    learner.train_model()
    learner.save_model("model_simple")


if __name__ == '__main__':
    run_ml_training()
