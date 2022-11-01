import argparse
import json
from pathlib import Path
import time

import numpy as np
import pescador
from yaml import load, Loader
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tqdm import tqdm

import models
import shared
import evaluate
from data_loaders import data_generator


def model_and_cost(config):
    model = models.classifier(config)

    if "classification" in config["task_type"]:
        loss = losses.CategoricalCrossentropy()
    elif config["task_type"] == "regression":
        loss = losses.MeanSquareError()
    else:
        raise Exception(f"task type {config['task_type']} is not defined!")

    if config["optimizer"] == "adam":
        optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
    else:
        raise Exception(f"optimizer {config['optimizer']} is not defined!")

    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="configuration file")
    parser.add_argument(
        "-s", "--single_batch", action="store_true", help="iterate over a single batch"
    )
    parser.add_argument(
        "-n", "--number_samples", type=int, help="iterate over a just n random samples"
    )

    args = parser.parse_args()

    config_file = args.config_file
    single_batch = args.single_batch
    number_samples = args.number_samples

    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
    exp_dir = Path(config["exp_dir"])
    data_dir = Path(config["data_dir"])

    np.random.seed(seed=config["seed"])

    # load audio representation paths
    file_index = data_dir / "index_repr.tsv"
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # load training data
    file_ground_truth_train = data_dir / "gt_train_0.csv"
    ids_train, id2gt_train = shared.load_id2gt(file_ground_truth_train)
    # load validation data
    file_ground_truth_val = data_dir / "gt_val_0.csv"
    ids_val, id2gt_val = shared.load_id2gt(file_ground_truth_val)

    print("# Train:", len(ids_train))
    print("# Val:", len(ids_val))
    print("# Classes:", config["n_classes"])

    # save experimental settings
    experiment_id = str(shared.get_epoch_time())
    model_folder = exp_dir / experiment_id
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(model_folder / "config.json", "w"))
    print("\nConfig file saved: " + str(config))

    # tensorflow: define model and cost
    model = model_and_cost(config)

    print("\nEXPERIMENT: ", str(experiment_id))
    print("-----------------------------------")

    if single_batch:
        print("Iterating over a single batch")

        if number_samples:
            size = number_samples
        else:
            size = config["batch_size"]
        np.random.seed(0)
        ids_train = list(
            np.array(ids_train)[np.random.randint(0, high=len(ids_train), size=size)]
        )
        ids_val = list(
            np.array(ids_val)[np.random.randint(0, high=len(ids_val), size=size)]
        )

        config["ids_train"] = ids_train
        config["ids_val"] = ids_val

        # Re-dump config with ids
        json.dump(config, open(model_folder + "config.json", "w"))

    n_batches_train = int(np.ceil(len(ids_train) / config["batch_size"]))
    n_batches_val = int(np.ceil(len(ids_val) / config["val_batch_size"]))
    # pescador train: define streamer
    train_pack = [config, config["train_sampling"], config["param_train_sampling"]]
    train_streams = [
        pescador.Streamer(
            data_generator, id, id2audio_repr_path[id], id2gt_train[id], train_pack
        )
        for id in ids_train
    ]
    train_mux_stream = pescador.StochasticMux(
        train_streams, n_active=config["batch_size"] * 2, rate=None, mode="exhaustive"
    )
    train_batch_streamer = pescador.Streamer(
        pescador.buffer_stream,
        train_mux_stream,
        buffer_size=config["batch_size"],
        partial=True,
    )
    train_batch_streamer = pescador.ZMQStreamer(train_batch_streamer)

    # pescador val: define streamer
    val_pack = [config, "overlap_sampling", config["x_size"]]
    val_streams = [
        pescador.Streamer(
            data_generator, id, id2audio_repr_path[id], id2gt_val[id], val_pack
        )
        for id in ids_val
    ]
    val_mux_stream = pescador.ChainMux(val_streams, mode="exhaustive")
    val_batch_streamer = pescador.Streamer(
        pescador.buffer_stream,
        val_mux_stream,
        buffer_size=config["val_batch_size"],
        partial=True,
    )
    val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)

    # writing headers of the train_log.tsv
    fy = open(model_folder / "train_log.tsv", "a")
    fy.write("Epoch\ttrain_cost\tval_cost\tepoch_time\tlearing_rate\n")
    fy.close()

    # training
    k_patience = 0
    cost_best_model = np.Inf
    tmp_learning_rate = config["learning_rate"]
    print("Training started..")

    for i in range(config["epochs"]):
        # modify the seed number on every epoch so that we get different patches
        # from each track
        np.random.seed(seed=config["seed"] + i)

        start_time = time.time()
        array_train_cost = []
        for batch in tqdm(train_batch_streamer, desc="train", total=n_batches_train):
            tf_start = time.time()

            train_cost = model.train_on_batch(batch["X"], batch["Y"])
            array_train_cost.append(train_cost)

        # validation
        array_val_cost = []
        for batch in tqdm(val_batch_streamer, desc="val"):
            val_cost = model.test_on_batch(batch["X"], batch["Y"])
            array_val_cost.append(val_cost)

        # Keep track of average loss of the epoch
        train_cost = np.mean(array_train_cost)
        val_cost = np.mean(array_val_cost)

        epoch_time = time.time() - start_time
        fy = open(model_folder / "train_log.tsv", "a")
        fy.write(
            "%g\t%g\t%g\t%gs\t%g\n"
            % (i + 1, train_cost, val_cost, epoch_time, tmp_learning_rate)
        )
        fy.close()

        # Decrease the learning rate after not improving in the validation set
        if config["patience"] and k_patience >= config["patience"]:
            print("Changing learning rate!")
            tmp_learning_rate = tmp_learning_rate / 2
            print(tmp_learning_rate)
            k_patience = 0
            K.set_value(model.optimizer.learning_rate, tmp_learning_rate)

        # Early stopping: keep the best model in validation set
        if val_cost >= cost_best_model:
            k_patience += 1
            print(
                "Epoch %d, train cost %g, "
                "val cost %g, "
                "epoch-time %gs, lr %g, time-stamp %s"
                % (
                    i + 1,
                    train_cost,
                    val_cost,
                    epoch_time,
                    tmp_learning_rate,
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
                )
            )

        else:
            # save model weights to disk
            model_path = model_folder / "checkpoint.hdf5"
            model.save(str(model_path))
            print(
                "Epoch %d, train cost %g, "
                "val cost %g, "
                "epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]"
                " saved in: %s"
                % (
                    i + 1,
                    train_cost,
                    val_cost,
                    epoch_time,
                    tmp_learning_rate,
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
                    model_path,
                )
            )
            cost_best_model = val_cost

    print("\nEVALUATE EXPERIMENT -> " + str(experiment_id))

    gt_file = data_dir / "gt_test_0.csv"
    print("groundtruth file: {}".format(gt_file))
    ids, id2gt = shared.load_id2gt(gt_file)
    print("# test set size: ", len(ids))

    print("performing regular evaluation")
    y_pred, metrics, healthy_ids = evaluate.prediction(
        config,
        model_folder,
        id2audio_repr_path,
        id2gt,
        ids,
    )

    # store experimental results
    results_file = Path(model_folder, "results.json")
    predictions_file = Path(model_folder, "predictions.json")
    evaluate.store_results(
        results_file,
        predictions_file,
        healthy_ids,
        y_pred,
        metrics,
    )
