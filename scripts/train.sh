#!/bin/bash
set -e

cd "$(dirname "$0")"

cfg_file=../cfg/config_effnet_mtg-jamendo-genre.yaml
python ../src/train.py "$cfg_file"
