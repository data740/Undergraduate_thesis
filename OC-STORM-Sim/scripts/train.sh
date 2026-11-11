#!/usr/bin/env bash
# 用法：
#   bash scripts/train.sh                                   # 默认：创新点 1/3/4 全开
#   CONFIG=atari_abl_none bash scripts/train.sh             # 原始 STORM
#   ENV=Breakout SEED=1 CONFIG=atari_abl_3 bash scripts/train.sh
# 可选 CONFIG：atari_visual(1+3+4) atari_abl_none atari_abl_1 atari_abl_3 atari_abl_4 atari_abl_13 atari_abl_14 atari_abl_34

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Boxing Pong etc. Gopher
env_prefix="${ENV:-Freeway}"
seed="${SEED:-46}"
config_name="${CONFIG:-atari_visual}"
run_name="${env_prefix}-${config_name}-seed${seed}"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --run_name "${run_name}" \
    --env_name "${env_prefix}NoFrameskip-v4" \
    --seed "${seed}" \
    --config_name "${config_name}"

######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"

# boss_name="MantisLords"
# python -u train_async.py \
#     --run_name "${boss_name}-async-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"

