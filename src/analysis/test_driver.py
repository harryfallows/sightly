import glob
import os
import pickle
from tensorflow.keras.models import load_model
import sys
sys.path.append("..")
from losses import categorical_focal_loss, binary_focal_loss
import dill
import argparse
import json
import pandas as pd


def main(model_folder, loss_flag, val_loss_flag):
    """Run tests on trained model.

    Arguments:
        model_folder {string} -- folder containing model
        loss_flag {bool} -- test weights saved at minimum loss during training
        val_loss_flag {bool} -- test weights saved at minimum validation loss during training
    """
    model_resource_info = json.load(
        open("{}/resource_info.json".format(model_folder), "rb")
    )
    system_folder = model_resource_info["system_folder"]
    preprocessed_data_folder = "{}/{}".format(
        system_folder, model_resource_info["preprocessed_data_folder"]
    )
    X_test = pickle.load(
        open("{}/X_test.pickle".format(preprocessed_data_folder), "rb")
    )
    y_test = pickle.load(
        open("{}/y_test.pickle".format(preprocessed_data_folder), "rb")
    )

    results_folder = "{}/test_results".format(model_folder)
    try:
        os.mkdir(results_folder)
    except FileExistsError:
        pass

    losses = []
    if loss_flag:
        losses.append("loss")
    if val_loss_flag:
        losses.append("val_loss")
    training_results_folder = "{}/output_weights".format(model_folder)
    training_test_results = pickle.load(
        open("{}/training_results.pickle".format(training_results_folder), "rb")
    )
    for loss_val_loss in losses:
        loss_results_folder = "{}/{}".format(results_folder, loss_val_loss)
        try:
            os.mkdir(loss_results_folder)
        except FileExistsError:
            pass
        model = load_model(
            model_folder + "/model.h5",
            custom_objects={
                "categorical_focal_loss": categorical_focal_loss,
                "categorical_focal_loss_fixed": dill.loads(
                    dill.dumps(categorical_focal_loss(gamma=2.0, alpha=0.25))
                ),
                "binary_focal_loss": binary_focal_loss,
                "binary_focal_loss_fixed": dill.loads(
                    dill.dumps(binary_focal_loss(gamma=2.0, alpha=0.25))
                ),
            },
        )
        list_of_files = glob.glob(
            "{}/output_weights/{}/*".format(model_folder, loss_val_loss)
        )
        latest_file = max(list_of_files, key=os.path.getctime)
        model.load_weights(latest_file)

        test_results = model.evaluate(
            x=X_test, y={k + "_o": v for k, v in y_test.items()}, verbose=1
        )
        training_test_results = training_test_results.append(
            pd.Series(
                test_results,
                [
                    "test_{}_{}".format(loss_val_loss, str(mm))
                    for mm in model.metrics_names
                ],
            )
        )
    pickle.dump(
        training_test_results,
        open("{}/test_results.pickle".format(results_folder), "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_folder",
        dest="model_folder",
        help="Folder containing model.",
        required=True,
    )
    parser.add_argument("-l", "--loss_flag", action="store_true", default=False)
    parser.add_argument("-v", "--val_loss_flag", action="store_true", default=False)

    args = parser.parse_args()
    model_folder = args.model_folder
    loss_flag = args.loss_flag
    val_loss_flag = args.val_loss_flag
    if not (loss_flag or val_loss_flag):
        loss_flag = True
        val_loss_flag = True
    if not os.path.isdir(model_folder):
        sys.exit("{} is not a directory".format(model_folder))
    main(model_folder, loss_flag, val_loss_flag)
