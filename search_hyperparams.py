"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
flags.DEFINE_string("model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("params_dir", default="./experiments/search_params", help="Path to model checkpoint (by default train from scratch).")

def launch_training_job(parent_dir, job_name, params):
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
     if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    path = FLAGS.dataset_path
    utils.check_dir(FLAGS.params_dir)
    params_path = os.path.join(FLAGS.model_dir, 'params.json')
    params = utils.Params(params_path)
    
    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(FLAGS.params_dir, job_name, params)
