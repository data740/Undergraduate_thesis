#!/usr/bin/env bash
# 用法（ENV / SEED / CONFIG 必须与训练时一致，才能找到检查点并构造相同的网络）：
#   bash scripts/eval.sh
#   ENV=Breakout SEED=1 CONFIG=atari_abl_3 bash scripts/eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Boxing Pong Breakout RoadRunner
env_prefix="${ENV:-Freeway}"
seed="${SEED:-46}"
config_name="${CONFIG:-atari_visual}"
run_name="${env_prefix}-${config_name}-seed${seed}"

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --save_frames false \
    --eval_episodes 100 \
    --run_name "${run_name}" \
    --env_name "${env_prefix}NoFrameskip-v4" \
    --seed "${seed}" \
    --config_name "${config_name}"


######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# eval
# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u eval.py \
#     --save_frames false \
#     --eval_episodes 10 \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"
