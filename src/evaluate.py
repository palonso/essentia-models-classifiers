import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tqdm import tqdm
import pescador

import train
import shared

TEST_BATCH_SIZE = 64


def prediction(
    config, experiment_folder, id2audio_repr_path, id2gt, ids, is_regression_task
):
    # pescador: define (finite, batched & parallel) streamer
    pack = [config, "overlap_sampling", config["x_size"]]
    streams = [
        pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt[id], pack)
        for id in ids
    ]
    mux_stream = pescador.ChainMux(streams, mode="exhaustive")
    batch_streamer = pescador.Streamer(
        pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True
    )
    batch_streamer = pescador.ZMQStreamer(batch_streamer)
    n_classes = config["n_classes"]

    # tensorflow: define model and cost
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        sess = tf.Session()
        [x, y_, is_train, y, normalized_y, cost, _] = train.tf_define_model_and_cost(
            config
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, str(experiment_folder) + "/")

        pred_list, id_list = [], []
        for batch in tqdm(batch_streamer):
            pred, _ = sess.run(
                [normalized_y, cost],
                feed_dict={x: batch["X"], y_: batch["Y"], is_train: False},
            )
            # make sure our predictions are in a numpy
            # array with the proper shape
            pred = np.array(pred).reshape(-1, n_classes)
            pred_list.append(pred)
            id_list.append(batch["ID"])

        pred_array = np.vstack(pred_list)
        id_array = np.hstack(id_list)

        sess.close()

    print(pred_array.shape)
    print(id_array.shape)

    print("Predictions computed, now evaluating...")
    y_true, y_pred, healthy_ids = shared.average_predictions(
        pred_array, id_array, ids, id2gt
    )

    if not is_regression_task:
        roc_auc, pr_auc = shared.compute_auc(y_true, y_pred)
        acc = shared.compute_accuracy(y_true, y_pred)

        metrics = (roc_auc, pr_auc, acc)
    else:
        # accuracy is only used for classification tasks
        pearson_corr = shared.compute_pearson_correlation(y_true, y_pred)
        ccc = shared.compute_ccc(y_true, y_pred)
        r2_score = shared.compute_r2_score(y_true, y_pred)
        adjusted_r2_score = shared.compute_adjusted_r2_score(
            y_true, y_pred, np.shape(y_true)[1]
        )
        rmse = shared.compute_root_mean_squared_error(y_true, y_pred)
        mse = shared.compute_mean_squared_error(y_true, y_pred)

        metrics = (pearson_corr, ccc, r2_score, adjusted_r2_score, rmse, mse)

    return y_pred, metrics, healthy_ids


def store_results(
    results_file, predictions_file, models, ids, y_pred, metrics, is_regression_task
):

    results_file.parent.mkdir(exist_ok=True, parents=True)

    if is_regression_task:
        pearson_corr, ccc, r2_score, adjusted_r2_score, rmse, mse = metrics

        # print experimental results
        print("Metrics:")
        print(f"PEARSONR: {pearson_corr}")
        print(f"CCC: {ccc}")
        print(f"R2 SCORE: {r2_score}")
        print(f"ADJUSTED R2 SCORE: {adjusted_r2_score}")
        print(f"RMSE: {rmse}")
        print(f"MSE: {mse}")

        to = open(results_file, "w")
        to.write("Experiment: " + str(models))
        to.write("\nPEARSONR: " + str(pearson_corr))
        to.write("\nCCC: " + str(ccc))
        to.write("\nR2 SCORE: " + str(r2_score))
        to.write("\nADJUSTED R2 SCORE: " + str(adjusted_r2_score))
        to.write("\nRMSE: " + str(rmse))
        to.write("\nMSE: " + str(mse))
        to.write("\n")
        to.close()
    else:
        roc_auc, pr_auc, acc = metrics

        # print experimental results
        print("Metrics:")
        print("ROC-AUC: " + str(roc_auc))
        print("PR-AUC: " + str(pr_auc))
        print("Acc: " + str(acc))

        to = open(results_file, "w")
        to.write("Experiment: " + str(models))
        to.write("\nROC AUC: " + str(roc_auc))
        to.write("\nPR AUC: " + str(pr_auc))
        to.write("\nAcc: " + str(acc))
        to.write("\n")
        to.close()

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
        config = json.load(f)
    config_train = config["config_train"]
    file_index = str(Path(config["data_dir"], "index_repr.tsv"))
    exp_dir = config["exp_dir"]

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    for model in models:
        experiment_folder = Path(exp_dir, "experiments", str(model))
        print("Experiment: " + str(model))
        print("\n" + str(config))

        feature_combination = "audio_representation_dirs" in config_train
        is_regression_task = config["config_train"]["task_type"] == "regression"

        # set patch parameters
        config_train["x_size"] = config_train["feature_params"]["x_size"]

        if feature_combination:
            config_train["y_size"] = sum(
                [i["y_size"] for i in config_train["features_params"]]
            )
        else:
            config_train["y_size"] = config_train["feature_params"]["y_size"]

        # get the data loader
        print("Loading data generator for regular training")
        if feature_combination:
            from data_loaders import data_gen_feature_combination as data_gen
        else:
            from data_loaders import data_gen_standard as data_gen

        # load ground truth
        print("groundtruth file: {}".format(config_train["gt_test"]))
        ids, id2gt = shared.load_id2gt(config_train["gt_test"])
        print("# Test set size: ", len(ids))

        print("Performing regular evaluation")
        y_pred, metrics, healthy_ids = prediction(
            config_train,
            experiment_folder,
            id2audio_repr_path,
            id2gt,
            ids,
            is_regression_task,
        )

        # store experimental results
        results_file = Path(exp_dir, f"results_{config_train['fold']}")
        predictions_file = Path(exp_dir, f"predictions_{config_train['fold']}.json")

        store_results(
            results_file,
            predictions_file,
            models,
            healthy_ids,
            y_pred,
            metrics,
            is_regression_task,
        )
