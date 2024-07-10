#!/bin/bash
CACHE=$1
OBJECT=$2
NAME=$3
SIZE=$4
python train.py task=AllegroHandRotateIt headless=False pipeline=gpu \
task.env.numEnvs=100 test=True \
task.env.object.type="${OBJECT}" \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
task.env.grasp_cache_name="${NAME}" \
task.env.grasp_cache_size="${SIZE}" \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth