"""把一次训练的 TensorBoard 日志汇总成一段文本，方便贴出来调参。

用法：
  python scripts/summarize_run.py runs/Freeway-atari_visual-seed46-260101-120000-0
  python scripts/summarize_run.py Freeway-atari_visual-seed46        # 给 run_name，自动取最新的日志目录
"""
import glob
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NUM_SEGMENTS = 10  # 把整个训练平均切成 10 段，每段取均值

TAGS = [
    ("sample/episode_return", "训练回合回报"),
    ("WorldModel/reconstruction_loss", "重建 loss"),
    ("WorldModel/reward_loss", "奖励 loss"),
    ("WorldModel/dynamics_real_kl_div", "真实 KL(后验||先验)，有输入偏差时会变大"),
    ("WorldModel/total_loss", "WM 总 loss"),
    ("WorldModel/grad_norm", "WM 梯度范数(裁剪阈值 1000)"),
    ("WorldModel/ss_mix_ratio", "[创新点1] 实际被替换的输入比例"),
    ("WorldModel/ss_kl_teacher_forced", "[创新点1] 全真实输入时的 KL"),
    ("WorldModel/ss_kl_mixed", "[创新点1] 混入预测输入后的 KL"),
    ("WorldModel/per_sample_loss_mean", "[创新点3] 窗口优先级 loss 均值(本批)"),
    ("Replay/wm_loss_mean", "[创新点3] 缓冲区内优先级均值"),
    ("Replay/wm_loss_max", "[创新点3] 缓冲区内优先级最大值"),
    ("Replay/untouched_ratio", "[创新点3] 仍为初值(从未被训练到)的窗口占比"),
    ("Replay/eff_size", "[创新点3] softmax 有效样本数 1/sum(p^2)，越小越集中"),
    ("WorldModel/prior_ensemble_loss", "[创新点4] K 个头的 loss"),
    ("WorldModel/prior_ensemble_jsd", "[创新点4] 训练批上的 JSD"),
    ("Imagine/intrinsic_reward_mean", "[创新点4] 想象中内在奖励均值"),
    ("Imagine/env_reward_abs_mean", "[创新点4] 想象中环境奖励绝对值均值"),
    ("ActorCritic/S", "回报分位差 S"),
    ("ActorCritic/norm_ratio", "max(1,S)"),
    ("ActorCritic/entropy_loss", "策略熵"),
]


def find_dir(arg):
    if os.path.isdir(arg):
        return arg
    cands = sorted(glob.glob(os.path.join("runs", arg + "-[0-9]*-[0-9]*-*")))
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        raise SystemExit(f"找不到日志目录: {arg}")
    return cands[-1]


def main():
    log_dir = find_dir(sys.argv[1])
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    print(f"log_dir: {log_dir}")
    print(f"每个指标把全程平均切成 {NUM_SEGMENTS} 段，依次给出各段均值（左=训练初期，右=末期）\n")
    for tag, desc in TAGS:
        if tag not in available:
            continue
        values = np.array([e.value for e in acc.Scalars(tag)], dtype=np.float64)
        if len(values) == 0:
            continue
        segs = [seg.mean() for seg in np.array_split(values, min(NUM_SEGMENTS, len(values)))]
        print(f"{tag}  ({desc}, n={len(values)})")
        print("  " + "  ".join(f"{v:.4g}" for v in segs))
    ret = np.array([e.value for e in acc.Scalars("sample/episode_return")]) if "sample/episode_return" in available else None
    if ret is not None and len(ret) >= 5:
        print(f"\n最近 5 个训练回合回报均值: {ret[-5:].mean():.2f}，全部回合均值: {ret.mean():.2f}，回合数: {len(ret)}")


if __name__ == "__main__":
    main()
