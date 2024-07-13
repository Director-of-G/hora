#!/bin/bash
CACHE=$1
python train.py task=AllegroHandRotateIt headless=False pipeline=gpu \
task.env.numEnvs=64 test=True task.on_evaluation=True \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandRotateIt/"${CACHE}" \
checkpoint=outputs/AllegroHandRotateIt/"${CACHE}"/stage1_nn/last.pth