# Essentia-models classifiers
This repository contains scripts to train and evaluate classifiers on top of audio representations (embeddings).

# Setup
1. We recommend following the official [instructions](https://www.tensorflow.org/install/pip) to install TensorFlow and the CUDA library with Conda.
2. After this, install the rest of dependencies in the created environment: `pip install -r requirements.txt`
2. Modify path to data in the configuration files by changing the `data_dir` field in the configuration files at `cfg/`

# Usage
This repository contains convenience scripts tho tran and evaluate the models.
To train and evaluate all the models execute `./train_all.sh` from the `scripts/` folder.
Additionally, the model incorporates a script `./evaluate.sh` to evaluate a pre-trained model. w

# Ground truth
The training and evaluation scripts expect the following files in the `data_dir`:
- gt_test_0.csv: tsv file containing the one-hot-encoded values of the testing samples: `id\t[0,1...0]`
- gt_train_0.csv: tsv file containing the one-hot-encoded values of the training samples: `id\t[0,1...0]`
- gt_val_0.csv: tsv file containing the one-hot-encoded values of the validation samples: `id\t[0,1...0]`
- index_repr.tsv: tsv file containing the correspondence between the filenames of the audio and representations: `audio_filename\trepresentation_filename`

# TODOs
- [ ] Fix freeze_model.py.
