#!/bin/bash
# CACHE can be some existing output folder, does not matter
# numEnvs=20000, headless=True, episodeLen=50 to save time
# see the object, check whether it's simple tennis ball or fancy balls
# pipeline need to be cpu to get the pairwise contact
# no custom PD because bug in CPU mode
# mass should be about 50g
GPUS=$1
SCALE=$2
ASSET=$3
SIZE=$4
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGrasp headless=True pipeline=cpu \
task.env.numEnvs=2000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=null \
task.env.object.objectAsset="${ASSET}" \
task.env.reset_height_threshold=0.635 \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
task.env.grasp_cache_name=null \
task.env.grasp_cache_size=${SIZE} \
train.ppo.priv_info=True