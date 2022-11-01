import numpy as np
import os
from pathlib import Path


def get_short_rep(audio_repr_path, x, y, frames_num):
    fp = np.memmap(audio_repr_path, dtype="float16", mode="r", shape=(frames_num, y))
    audio_rep = np.zeros([x, y])
    audio_rep[:frames_num, :] = np.array(fp)
    del fp

    return audio_rep


def read_mmap(audio_repr_path, x, y, frames_num, single_patch=False, offset=0):
    if frames_num < x:
        audio_repr = get_short_rep(audio_repr_path, x, y, frames_num)
    else:
        read_x = x if single_patch else frames_num
        fp = np.memmap(
            audio_repr_path, dtype="float16", mode="r", shape=(read_x, y), offset=offset
        )
        audio_repr = np.array(fp)
        del fp
    return audio_repr


def data_generator(id, audio_repr_path, gt, pack):
    config, sampling, param_sampling = pack
    audio_repr_path = Path(
        config["data_dir"], f"{config['dataset']}__time-freq", audio_repr_path
    )

    try:
        floats_num = os.path.getsize(audio_repr_path) // 2  # each float16 has 2 bytes
        frames_num = floats_num // config["y_size"]

        # let's deliver some data!
        if sampling == "random":
            for i in range(0, param_sampling):
                # we use a uniform distribution to get a relative random offset depending
                # exclusively in the seed number and not in the number of frames.
                # This way for two feature types with different number of frames the
                # sampler will select roughly the same chunks of the audio.
                random_uniform = np.random.random()
                random_frame_offset = int(
                    round(random_uniform * (frames_num - config["x_size"]))
                )

                # idx * bands * bytes per float
                offset = random_frame_offset * config["y_size"] * 2
                representation = read_mmap(
                    audio_repr_path,
                    config["x_size"],
                    config["y_size"],
                    frames_num,
                    single_patch=True,
                    offset=offset,
                )

                # flatten the temporal axis
                representation = representation.flatten()

                yield {
                    "X": representation,
                    "Y": gt,
                    "ID": id,
                }

        elif sampling == "overlap_sampling":
            audio_rep = read_mmap(
                audio_repr_path,
                config["x_size"],
                config["y_size"],
                frames_num,
            )
            last_frame = int(audio_rep.shape[0]) - int(config["x_size"]) + 1
            for time_stamp in range(0, last_frame, param_sampling):
                representation = audio_rep[
                    time_stamp : time_stamp + config["x_size"], :
                ]
                representation = representation.flatten()
                yield {
                    "X": representation,
                    "Y": gt,
                    "ID": id,
                }
    except FileNotFoundError:
        print('"{}" not found'.format(audio_repr_path))
