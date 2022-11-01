#!/bin/bash
set -e

cd "$(dirname "$0")"

for model in effnet vggish
do
    for subset in genre instrument moodtheme
    do
        echo "training embeddings ($model), in subset ($subset)"
        python ../src/train.py ../cfg/config_"$model"_mtg-jamendo-"$subset".yaml
    done
done
