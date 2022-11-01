import argparse
import json
from pathlib import Path

import numpy as np
import pescador
from yaml import load, Loader
from tqdm import tqdm

import train
from data_loaders import data_generator
import shared


def prediction(
    config,
    experiment_folder,
    id2audio_repr_path,
    id2gt,
    ids,
):
    # pescador: define (finite, batched & parallel) streamer
    pack = [config, "overlap_sampling", config["x_size"]]
    streams = [
        pescador.Streamer(data_generator, id, id2audio_repr_path[id], id2gt[id], pack)
        for id in ids
    ]
    mux_stream = pescador.ChainMux(streams, mode="exhaustive")
    batch_streamer = pescador.Streamer(
        pescador.buffer_stream,
        mux_stream,
        buffer_size=config["val_batch_size"],
        partial=True,
    )
    batch_streamer = pescador.ZMQStreamer(batch_streamer)

    # tensorflow: define model and cost
    model = train.model_and_cost(config)
    model.load_weights(experiment_folder / "checkpoint.hdf5")

    pred_list, id_list = [], []
    for batch in tqdm(batch_streamer):
        predictions = model.predict_on_batch(batch["X"])
        # make sure our predictions are in a numpy
        # array with the proper shape
        pred_list.append(predictions)
        id_list.append(batch["ID"])

    pred_array = np.vstack(pred_list)
    id_array = np.hstack(id_list)

    print(f"predictions shape: {pred_array.shape}")
    print(f"ID shape: {id_array.shape}")

    print("Predictions computed, now evaluating...")
    y_true, y_pred, healthy_ids = shared.average_predictions(
        pred_array, id_array, ids, id2gt
    )

    metrics = dict()
    if config["task_type"] == "multi-class-classification":
        metrics["accuracy"] = shared.compute_accuracy(y_true, y_pred)

    elif config["task_type"] == "multi-label-classification":
        roc_auc, pr_auc = shared.compute_auc(y_true, y_pred)
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc

    elif config["task_type"] == "regression":
        config["pearson_corr"] = shared.compute_pearson_correlation(y_true, y_pred)
        config["ccc"] = shared.compute_ccc(y_true, y_pred)
        config["r2_score"] = shared.compute_r2_score(y_true, y_pred)
        config["adjusted_r2_score"] = shared.compute_adjusted_r2_score(
            y_true, y_pred, np.shape(y_true)[1]
        )
        config["rmse"] = shared.compute_root_mean_squared_error(y_true, y_pred)
        config["mse"] = shared.compute_mean_squared_error(y_true, y_pred)
    else:
        raise (NotImplementedError("task type not defined"))
    return y_pred, metrics, healthy_ids


def store_results(results_file, predictions_file, ids, y_pred, metrics):
    results_file.parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, "w") as rfile:
        json.dump(metrics, rfile)

    lines = [f"{k}: {v:.3f}" for k, v in metrics.items()]
    print("\n".join(lines))

    predictions = {id: list(pred.astype("float64")) for id, pred in zip(ids, y_pred)}
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    # which experiment we want to evaluate?
    # Use the -l functionality to ensemble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="configuration file")
    parser.add_argument(
        "-l", "--list", nargs="+", help="List of models to evaluate", required=True
    )

    args = parser.parse_args()

    models = args.list
    config_file = Path(args.config_file)

    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)

    file_index = str(Path(config["data_dir"], "index_repr.tsv"))
    exp_dir = Path(config["exp_dir"])
    data_dir = Path(config["data_dir"])

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    for model in models:
        experiment_folder = Path(exp_dir, "experiments", str(model))
        print("Experiment: " + str(model))
        print("\n" + str(config))

        # load ground truth
        gt_file = data_dir / "gt_test_0.csv"
        print("groundtruth file: {}".format(gt_file))
        ids, id2gt = shared.load_id2gt(gt_file)
        print("# Test set size: ", len(ids))

        print("Performing regular evaluation")
        y_pred, metrics, healthy_ids = prediction(
            config,
            experiment_folder,
            id2audio_repr_path,
            id2gt,
            ids,
        )

        # store experimental results
        results_file = Path(exp_dir, "results.json")
        predictions_file = Path(exp_dir, "predictions.json")
        store_results(
            results_file,
            predictions_file,
            healthy_ids,
            y_pred,
            metrics,
        )
