#!/bin/bash
set -e

cd "$(dirname "$0")"


cfg_file=../cfg/config_file.yaml
python ../src/evaluate.py "$cfg_file" -l "$1"
