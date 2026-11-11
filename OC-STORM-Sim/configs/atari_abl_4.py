# 只开创新点 4；其余参数沿用 atari_visual.py 的默认值
from configs.atari_visual import Params, build_with

OVERRIDES = dict(scheduled_sampling_prob=0.0, wm_priority_fraction=0.0)


def build(env_name, seed):
    return build_with(env_name, seed, **OVERRIDES)
