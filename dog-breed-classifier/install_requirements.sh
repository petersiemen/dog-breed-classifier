#!/usr/bin/env bash

conda install --file conda_requirements.txt

pip install -r requirements.txt

conda install pytorch-cpu torchvision-cpu -c pytorch


