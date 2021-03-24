import argparse
import yaml

from . import enable_determinism as dt
from . import train


def run_ml_training(hyper_param=dict(), output_dir='models/'):
    """
    Function to run the machine learning model
    """
    # evaluate model with single split
    learner = train.Learner(output_dir)
    learner.train_model(hyper_param)
    learner.save_model("model_simple")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hyperparam", help="a YAML file specifying hyper-parameters")
    parser.add_argument("-o", "--output-dir", help="a output directory to store models")
    args = parser.parse_args()

    out_dir = args.output_dir
    if not out_dir:
        out_dir = "models/"

    if args.hyperparam is None:
        run_ml_training(output_dir=out_dir)
        exit()

    out_dir = args.output_dir
    if not out_dir:
        out_dir="models/"

    print(out_dir)
    with open(args.hyperparam) as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
        seed = hp.get('seed')
        full_determinism = hp.get('full_determinism', False)
        if seed:
            print("Setting up a random seed:" + str(seed))
            dt.set_global_determinism(seed=seed, fast_n_close=full_determinism)

        #tracking = hp.get('tracking_enabled', False)
        #if tracking:
        #    import mlflow
        #    mlflow.tensorflow.autolog()
        #    mlflow.log_params(hp)

        run_ml_training(hp, output_dir=out_dir)
