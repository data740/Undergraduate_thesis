# OC-STORM-Sim：创新点 1、3、4 的实现说明

在 `OC-STORM-Sim`（纯视觉 STORM，`agents/visual/`）中实现三个互相独立、可单独开关的模块：

- 创新点 1：世界模型训练时，以概率 p 把输入的 `z_t` 换成模型自己预测的 `z_t^predict`（scheduled sampling）。
- 创新点 3：世界模型训练批次的一部分按 WM loss 优先级采样（Simulus 的 curious replay）。
- 创新点 4：K 个先验头 + 分布分歧（JSD）作为想象中的内在奖励（Simulus 的 ensemble curiosity）。

创新点 2（多步预测 `z_{t+2..t+K}`）未实现。

## 0. 参数（`configs/atari_visual.py`）

| 参数 | 默认值 | 属于 | 含义 |
|---|---|---|---|
| `scheduled_sampling_prob` | 0.3 | 创新点 1 | 输入 `z_t` 被替换成预测值的概率，0 关闭 |
| `wm_priority_fraction` | 0.3 | 创新点 3 | 每个 WM batch 中按 loss 优先级取样的比例，0 关闭 |
| `wm_priority_init_loss` | 10.0 | 创新点 3 | 新数据的初始优先级 loss |
| `wm_priority_temperature` | 1.0 | 创新点 3 | `softmax(loss / T)` 的温度，1.0 即 Simulus 的做法，越大越平 |
| `num_prior_heads` | 4 | 创新点 4 | 先验头个数 K，0 关闭 |
| `intrinsic_reward_weight` | 1.0 | 创新点 4 | 内在奖励权重 |

`wm_priority_fraction=0.3` 对应 Simulus 的 `replay_sampling_uniform_fraction=0.7`，`init_loss=10` 对应 `CuriousReplayDistribution._loss_init_value=10`，`K=4` 对应 `curiosity_ensemble_size: 4`。`scheduled_sampling_prob=0.3` 来自 `innovation.md`。`wm_priority_temperature` 是我新增的参数，Simulus 没有。

## 1. 创新点 1：训练时混入预测的 `z_t`

### 1.1 做法

STORM 原来的世界模型训练全程 teacher forcing：Transformer 在每个位置 t 读真实后验采样 `z_t`，预测 `z_{t+1}`。想象阶段则读自己预测的 `z_t`，误差会累积，但训练时从没见过这种输入。

实现分两次前向（都在 `WorldModel.update` 里）：

1. 第一次前向在 `no_grad` 下用全真实输入，得到每步的先验 `prior(h_{t-1})`，从中采样出 `z_t^predict`（t = 1..L-1）。
2. 对 t = 1..L-1 的每个位置，以概率 p 把输入 `z_t` 换成 `z_t^predict`（`z_0` 始终是真实的）。第二次前向读混合后的输入，**所有 loss 都来自第二次前向**。

各项 loss 的监督目标不变：后验 `z_{t+1}`（来自真实观测编码）、真实奖励、真实终止信号、重建的真实图像。也就是说，只改 Transformer 的输入，不改目标。

### 1.2 代码：`agents/visual/world_model.py`

**原代码（`WorldModel.update`）：**

```python
            temporal_mask = get_causal_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
```

**改动后：**

```python
            temporal_mask = get_causal_mask_with_batch_length(batch_length, flattened_sample.device)
            transformer_input = flattened_sample
            if self.scheduled_sampling_prob > 0:
                # pass 1 全真实输入、不求导，得到 z_t^predict ~ prior(h_{t-1})
                with torch.no_grad():
                    teacher_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
                    teacher_prior_logits = self.dist_head.forward_prior(teacher_feat)
                    predicted_sample = self.flatten_sample(
                        self.straight_through_gradient(teacher_prior_logits[:, :-1], sample_mode="random_sample")
                    )
                    use_predicted = torch.rand(batch_size, batch_length - 1, 1, device=flattened_sample.device)
                    use_predicted = use_predicted < self.scheduled_sampling_prob
                    mixed_tail = torch.where(
                        use_predicted, predicted_sample.to(flattened_sample.dtype), flattened_sample[:, 1:]
                    )
                    transformer_input = torch.cat([flattened_sample[:, :1], mixed_tail], dim=1)  # z_0 始终真实
            # pass 2：之后所有 loss 都来自这次前向
            dist_feat = self.storm_transformer(transformer_input, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
```

构造函数新增参数 `scheduled_sampling_prob=0.0`，存为 `self.scheduled_sampling_prob`；`agents/visual/agent.py` 的 `Agent.__init__` 同名参数透传给 `WorldModel`；`configs/atari_visual.py` 的 `Params` 加 `scheduled_sampling_prob: float = 0.3` 并在 `build` 里传给 `Agent`（见第 4.5 节）。

新增日志（`update` 末尾）：

```python
            if self.scheduled_sampling_prob > 0:
                with torch.no_grad():
                    post_next = OneHotCategorical(logits=post_logits[:, 1:].detach().float())
                    kl_teacher = torch.distributions.kl.kl_divergence(
                        post_next, OneHotCategorical(logits=teacher_prior_logits[:, :-1].float())
                    ).sum(dim=-1)
                    kl_mixed = torch.distributions.kl.kl_divergence(
                        post_next, OneHotCategorical(logits=prior_logits[:, :-1].detach().float())
                    ).sum(dim=-1)
                    logger.log("WorldModel/ss_mix_ratio", use_predicted.float().mean().item())
                    logger.log("WorldModel/ss_kl_teacher_forced", kl_teacher.mean().item())
                    logger.log("WorldModel/ss_kl_mixed", kl_mixed.mean().item())
```

`ss_kl_teacher_forced` 是第一次（全真实输入）前向的真实 KL，`ss_kl_mixed` 是混入后第二次前向的真实 KL，两者都不含 free bits。

### 1.3 要注意的实现细节

- 每个训练步多一次 Transformer 前向（不求导）。我没有测过实际耗时增加了多少。
- `z_t^predict` 来自**用真实 `z_{<t}` 算出的** `h_{t-1}`，所以它是"一步预测"，不是完整的自回归展开。第二次前向里，被替换位置之后的 `z` 是真实的，被替换的位置之间不会级联。想象阶段的误差累积只被部分模拟。
- 与创新点 4 同时开启时，先验头的输入 `h_t` 来自混合输入的第二次前向。

## 2. 创新点 3：按 WM loss 优先级采样

### 2.1 Simulus 的做法（参考）

`Simulus/src/dataset.py` 的 `CuriousReplayDistribution.sample`：

```python
prioritized_bsz = math.ceil((1 - self.uniform_portion) * batch_size)
dist = self.distribution                      # softmax(self.losses)
prioritized_sample = torch.multinomial(dist, prioritized_bsz, replacement=False)
p = torch.ones_like(dist)
p[prioritized_sample] = 0
p = p / p.sum()
uniform_sample = torch.multinomial(p, batch_size - prioritized_bsz, replacement=False)
sample = torch.cat([prioritized_sample, uniform_sample], dim=0)
sample = sample[torch.randperm(sample.size(0), device=self.device)]  # shuffle so that curiosity ensemble members will get random splits
```

新样本的 loss 初值为 `_loss_init_value = 10`；每步训练后用 `replay_dist.update_losses(info['per_sample_loss'])` 回写。

### 2.2 `utils/replay_buffer.py`（`VisualReplayBuffer`）

**原代码：**

```python
class VisualReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        ...
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        ...

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])
        ...
        obs = obs.to(torch.float32) / 255

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination, episode):
        ...
        self.episode_buffer[self.last_pointer] = episode
```

**改动后：**

```python
class VisualReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_dim,
        num_envs,
        max_length=int(1e6),
        warmup_length=50000,
        store_on_gpu=False,
        wm_priority_fraction=0.0,          # 新增
        wm_priority_init_loss=10.0,        # 新增
        wm_priority_temperature=1.0,       # 新增
    ) -> None:
        ...
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        # wm_loss_buffer[i] = 以 i 为起点的窗口最近一次的世界模型 loss，新数据从 init_loss 开始
        self.wm_priority_fraction = wm_priority_fraction
        self.wm_priority_init_loss = wm_priority_init_loss
        self.wm_priority_temperature = wm_priority_temperature
        self.wm_loss_buffer = torch.full((max_length,), wm_priority_init_loss, dtype=torch.float32, device="cuda")

        self.length = 0
        ...

    def _gather(self, indices, batch_length):          # 原 sample 的取数部分，原样抽出
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])
        ...
        obs = obs.to(torch.float32) / 255

        return obs, action, reward, termination

    @torch.no_grad()
    def sample(self, batch_size, batch_length):        # 行为不变
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return self._gather(indices, batch_length)

    @torch.no_grad()
    def sample_wm(self, batch_size, batch_length):     # 新增：世界模型训练专用
        assert batch_size > 0, "batch_size must be greater than 0"

        sample_limit = self.length + 1 - batch_length
        num_prioritized = math.ceil(self.wm_priority_fraction * batch_size)
        num_prioritized = min(num_prioritized, batch_size, sample_limit)

        parts = []
        if num_prioritized > 0:
            prob = torch.softmax(self.wm_loss_buffer[:sample_limit] / self.wm_priority_temperature, dim=0)
            parts.append(torch.multinomial(prob, num_prioritized, replacement=False).cpu().numpy())
        if batch_size - num_prioritized > 0:
            parts.append(self.sample_indices(batch_size - num_prioritized, sample_limit))  # 其余走 STORM 原采样器
        indices = np.concatenate(parts)
        # 打乱，使每个先验头（batch 的一个分块）拿到随机混合的优先级 / 普通样本
        indices = indices[np.random.permutation(len(indices))]

        return (*self._gather(indices, batch_length), indices)

    @torch.no_grad()
    def wm_priority_stats(self):                       # 新增：调参用诊断
        stored = self.wm_loss_buffer[: self.length]
        prob = torch.softmax(stored / self.wm_priority_temperature, dim=0)
        return {
            "wm_loss_mean": stored.mean().item(),
            "wm_loss_max": stored.max().item(),
            "untouched_ratio": (stored == self.wm_priority_init_loss).float().mean().item(),
            "eff_size": (1.0 / (prob**2).sum()).item(),
        }

    @torch.no_grad()
    def update_wm_priorities(self, indices, losses):   # 新增：回写
        indices = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        self.wm_loss_buffer[indices] = losses.detach().to(torch.float32)

    def append(self, obs, action, reward, termination, episode):
        ...
        self.episode_buffer[self.last_pointer] = episode
        self.wm_loss_buffer[self.last_pointer] = self.wm_priority_init_loss   # 新增：覆盖旧数据时重置优先级
```

对应关系：Simulus 的"30% 按 softmax(loss) 无放回抽、70% 均匀"在这里变成"30% 按 softmax(loss / T) 无放回抽、70% 走 STORM 原来的 `sample_indices`（近期衰减 + 均匀混合）"。

### 2.3 `agents/visual/modules/functions_losses.py`

逐样本的 reward loss 需要不聚合的版本。

**原代码：**

```python
    def forward(self, output, target):
        ...
        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()
```

**改动后：**

```python
    def forward(self, output, target, reduction="mean"):
        ...
        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        if reduction == "none":
            return loss
        return loss.mean()
```

### 2.4 `agents/visual/world_model.py`：`update` 返回逐窗口 loss

**原代码：**

```python
    def update(self, state, action, reward, termination, logger=None):
        ...
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        ...
            logger.log("WorldModel/total_loss", total_loss.item())
```

**改动后：**

```python
    def update(self, state, action, reward, termination, logger=None, return_per_sample_loss=False):
        ...
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )
            ...
            per_sample_loss = None
            if return_per_sample_loss:
                # 窗口优先级 = 下一步潜变量 KL（不含 free bits）+ reward loss，沿时间取平均
                with torch.no_grad():
                    kl = torch.distributions.kl.kl_divergence(
                        OneHotCategorical(logits=post_logits[:, 1:].detach().float()),
                        OneHotCategorical(logits=prior_logits[:, :-1].detach().float()),
                    ).sum(dim=-1)  # [B, L-1]
                    reward_per_step = self.symlog_twohot_loss_func(reward_hat.detach(), reward, reduction="none")
                    per_sample_loss = kl.mean(dim=1) + reward_per_step.float().mean(dim=1)

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)   # 现在保留返回值用于日志
        ...
            logger.log("WorldModel/total_loss", total_loss.item())
            logger.log("WorldModel/grad_norm", float(grad_norm))
            if per_sample_loss is not None:
                logger.log("WorldModel/per_sample_loss_mean", per_sample_loss.mean().item())

        return per_sample_loss
```

### 2.5 `agents/visual/agent.py`：训练循环

**原代码：**

```python
        self.world_model.train()
        obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length)
        self.world_model.update(obs, action, reward, termination, logger=logger)
```

**改动后：**

```python
        self.world_model.train()
        if replay_buffer.wm_priority_fraction > 0:
            obs, action, reward, termination, indices = replay_buffer.sample_wm(batch_size, batch_length)
            per_sample_loss = self.world_model.update(
                obs, action, reward, termination, logger=logger, return_per_sample_loss=True
            )
            replay_buffer.update_wm_priorities(indices, per_sample_loss)
            if self.update_steps % 100 == 0:  # 调参诊断
                for name, value in replay_buffer.wm_priority_stats().items():
                    logger.log(f"Replay/{name}", value)
        else:
            obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length)
            self.world_model.update(obs, action, reward, termination, logger=logger)
```

### 2.6 与 Simulus 的差异

- Simulus 的逐样本 loss 是下一帧 token 的交叉熵加奖励损失。STORM 没有 token，所以用"下一步潜变量 KL + 奖励 loss"代替，不含重建项。
- Simulus 的优先级对象是数据集里的样本；这里是回放缓冲区里以每个位置为起点的窗口，用定长平铺数组存储，没有用 Simulus 的 priority tree。
- 窗口之间重叠，一个窗口被抽到后只更新它起点的优先级，相邻起点的优先级不变。
- 窗口可能跨 episode 边界（STORM 原有行为）。

## 3. 创新点 4：K 个先验头 + JSD 内在奖励

### 3.1 Simulus 的做法（参考）

`Simulus/src/envs/world_model_env.py::_sample_obs_and_rewards`：

```python
rewards, ends = self.world_model.sample_rewards_ends(preds)
if self.world_model.enable_curiosity:
    ...
    intrinsic_reward = torch.cat([
        self.world_model.curiosity_head[m.name].estimate_uncertainty(preds[i])[0].mean(-1, keepdim=True)
        for i, m in enumerate(self.world_model.ordered_modalities)
    ], dim=-1).sum(dim=-1, keepdim=True)
    rewards = self.real_reward_weight * rewards + self.intrinsic_reward_weight * intrinsic_reward
```

集成头在 detach 后的主干输出上训练，每个头只用 batch 的 1/K 分块；不确定性是 K 个预测分布的 JSD（`H(mean p) − mean H(p)`）。

### 3.2 `agents/visual/world_model.py`：新增 `EnsemblePriorHead`

**原代码：** 无（新增类，放在 `DistHead` 之后）。

**改动后：**

```python
class EnsemblePriorHead(nn.Module):
    def __init__(self, transformer_hidden_dim, stoch_dim, ensemble_size) -> None:
        super().__init__()
        assert ensemble_size >= 2, "need at least 2 heads to measure disagreement"
        self.stoch_dim = stoch_dim
        self.ensemble_size = ensemble_size
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(transformer_hidden_dim, transformer_hidden_dim * 2),
                    nn.LayerNorm(transformer_hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(transformer_hidden_dim * 2, stoch_dim * stoch_dim),
                )
                for _ in range(ensemble_size)
            ]
        )

    def _logits(self, head, x):
        logits = head(x)
        return einops.rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)

    def forward_all(self, x):
        # x: [B, L, H] -> [K, B, L, stoch_dim, stoch_dim]
        return torch.stack([self._logits(head, x) for head in self.heads])

    def loss(self, feat, target_logits):
        """feat: [B, L, H] hidden state h_t, target_logits: [B, L, stoch_dim, stoch_dim] posterior of z_{t+1}."""
        feat = feat.detach()                                         # 梯度不进入 WM 主干
        target_probs = F.softmax(target_logits.detach().float(), dim=-1)
        chunks = torch.tensor_split(torch.arange(feat.shape[0], device=feat.device), self.ensemble_size)
        total = feat.new_zeros((), dtype=torch.float32)
        for head, idx in zip(self.heads, chunks):                    # 第 k 个头只用第 k 份 batch
            log_prob = F.log_softmax(self._logits(head, feat[idx]).float(), dim=-1)
            ce = -(target_probs[idx] * log_prob).sum(dim=-1)         # [b, L, stoch_dim]
            total = total + ce.sum(dim=-1).mean()
        return total

    @torch.no_grad()
    def estimate_uncertainty(self, feat):
        """K 个头的 JSD，对 32 个潜变量取平均 -> [B, L]"""
        log_p = F.log_softmax(self.forward_all(feat).float(), dim=-1)  # [K, B, L, S, C]
        p = log_p.exp()
        mean_p = p.mean(dim=0)
        entropy_of_mean = -(mean_p * torch.log(mean_p.clamp_min(1e-8))).sum(dim=-1)
        mean_of_entropy = -(p * log_p).sum(dim=-1).mean(dim=0)
        jsd = (entropy_of_mean - mean_of_entropy).clamp_min(0.0)
        return jsd.mean(dim=-1)
```

### 3.3 `agents/visual/world_model.py`：构造函数、训练 loss、想象奖励

**原代码：**

```python
class WorldModel(nn.Module):
    def __init__(self, ..., transformer_num_heads):
        ...
        self.termination_decoder = TerminationDecoder(
            embedding_size=1024, transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
```

```python
    def imagine_data(...):
        ...
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i : i + 1], self.action_buffer[:, i : i + 1], log_video=log_video
            )

            self.latent_buffer[:, i + 1 : i + 2] = last_latent
            self.hidden_buffer[:, i + 1 : i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i : i + 1] = last_reward_hat
```

```python
    def update(...):
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )
```

**改动后：**

```python
class WorldModel(nn.Module):
    def __init__(self, ..., transformer_num_heads, num_prior_heads=0, intrinsic_reward_weight=1.0,
                 scheduled_sampling_prob=0.0):
        ...
        self.termination_decoder = TerminationDecoder(
            embedding_size=1024, transformer_hidden_dim=transformer_hidden_dim
        )
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.last_intrinsic_reward_mean = 0.0
        self.last_real_reward_abs_mean = 0.0
        self.prior_ensemble = (
            EnsemblePriorHead(transformer_hidden_dim, stoch_dim=32, ensemble_size=num_prior_heads)
            if num_prior_heads > 0
            else None
        )

        self.mse_loss_func = MSELoss()
```

```python
    def imagine_data(...):
        obs_hat_list = []
        intrinsic_reward_sum = 0.0
        real_reward_abs_sum = 0.0
        ...
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i : i + 1], self.action_buffer[:, i : i + 1], log_video=log_video
            )

            if self.prior_ensemble is not None:
                # K 个头对 z_{t+1} 的分歧作为探索奖励
                with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
                    intrinsic_reward = self.prior_ensemble.estimate_uncertainty(last_dist_feat)  # [B, 1]
                intrinsic_reward_sum = intrinsic_reward_sum + intrinsic_reward.mean()
                real_reward_abs_sum = real_reward_abs_sum + last_reward_hat.abs().float().mean()
                last_reward_hat = last_reward_hat + self.intrinsic_reward_weight * intrinsic_reward

            self.latent_buffer[:, i + 1 : i + 2] = last_latent
            self.hidden_buffer[:, i + 1 : i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i : i + 1] = last_reward_hat      # actor-critic 实际使用的奖励
        ...
        if self.prior_ensemble is not None:
            self.last_intrinsic_reward_mean = float(intrinsic_reward_sum) / imagine_batch_length
            self.last_real_reward_abs_mean = float(real_reward_abs_sum) / imagine_batch_length
```

```python
    def update(...):
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )
            if self.prior_ensemble is not None:
                # h_t -> z_{t+1}，输入在内部 detach，主干不受影响
                prior_ensemble_loss = self.prior_ensemble.loss(dist_feat[:, :-1], post_logits[:, 1:])
                total_loss = total_loss + prior_ensemble_loss
        ...
            if self.prior_ensemble is not None:
                logger.log("WorldModel/prior_ensemble_loss", prior_ensemble_loss.item())
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp
                ):
                    jsd = self.prior_ensemble.estimate_uncertainty(dist_feat[:, :-1].detach())
                logger.log("WorldModel/prior_ensemble_jsd", jsd.mean().item())
```

### 3.4 `agents/visual/agent.py`：参数透传与日志

**原代码：**

```python
        entropy_coef,
    ) -> None:
        ...
            transformer_num_heads=transformer_num_heads,
        )
```

**改动后：**

```python
        entropy_coef,
        num_prior_heads=0,
        intrinsic_reward_weight=1.0,
        scheduled_sampling_prob=0.0,
    ) -> None:
        ...
            transformer_num_heads=transformer_num_heads,
            num_prior_heads=num_prior_heads,
            intrinsic_reward_weight=intrinsic_reward_weight,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
```

```python
            # 想象之后
            if self.world_model.prior_ensemble is not None:
                logger.log("Imagine/intrinsic_reward_mean", self.world_model.last_intrinsic_reward_mean)
                logger.log("Imagine/env_reward_abs_mean", self.world_model.last_real_reward_abs_mean)
```

### 3.5 `configs/atari_visual.py`（三个创新点共用）

**原代码：**

```python
    batch_size: int = 32
    batch_length: int = 32

    imagine_batch_size: int = 512
```

```python
def build(env_name, seed) -> Tuple[Params, Env, ActionSpace, ResizeObservation, VisualReplayBuffer, Agent]:
    params = Params()
    ...
    replay_buffer = VisualReplayBuffer(
        ...
        store_on_gpu=True,
    )
    agent = Agent(
        ...
        entropy_coef=params.policy.entropy_coef,
    ).cuda()

    return params, env, action_space, feature_extractor, replay_buffer, agent
```

**改动后：**

```python
    batch_size: int = 32
    batch_length: int = 32

    # innovation 1
    scheduled_sampling_prob: float = 0.3
    # innovation 3
    wm_priority_fraction: float = 0.3
    wm_priority_init_loss: float = 10.0
    wm_priority_temperature: float = 1.0
    # innovation 4
    num_prior_heads: int = 4
    intrinsic_reward_weight: float = 1.0

    imagine_batch_size: int = 512
```

```python
def build_with(env_name, seed, **overrides) -> Tuple[Params, Env, ActionSpace, ResizeObservation, VisualReplayBuffer, Agent]:
    params = Params(**overrides)          # 新增：允许覆盖参数，消融配置用
    ...
    replay_buffer = VisualReplayBuffer(
        ...
        store_on_gpu=True,
        wm_priority_fraction=params.wm_priority_fraction,
        wm_priority_init_loss=params.wm_priority_init_loss,
        wm_priority_temperature=params.wm_priority_temperature,
    )
    agent = Agent(
        ...
        entropy_coef=params.policy.entropy_coef,
        num_prior_heads=params.num_prior_heads,
        intrinsic_reward_weight=params.intrinsic_reward_weight,
        scheduled_sampling_prob=params.scheduled_sampling_prob,
    ).cuda()

    return params, env, action_space, feature_extractor, replay_buffer, agent


def build(env_name, seed):                # 原 build 的入口保持不变，train.py / eval.py 仍然调它
    return build_with(env_name, seed)
```

### 3.6 与 `innovation.md` 原文、Simulus 的差异

- "根据 h_t 输出 K 个先验头，只前向传递，不传递到 wm 主干"：实现为输入 `detach`，头本身被训练，梯度不回传主干。
- "分布差异作为额外奖励加到环境奖励"：实现为 JSD 加到想象中的预测奖励（`reward_hat_buffer`），不影响真实环境交互，也不进回放缓冲区。
- `innovation.md` 中"走对得分有用的区域"这一点**没有实现**，当前只是纯分歧奖励。Simulus 里也没有这个约束。
- Simulus 的头预测下一帧 token，这里预测 STORM 的 32×32 离散潜变量；Simulus 对多个模态求和，STORM 只有一个模态，直接对 32 个潜变量取平均。
- Simulus 有 `real_reward_weight` 缩放真实奖励，这里没有加，真实奖励保持原样。
- 先验头与主干共用同一个 Adam（`lr=1e-4`），ensemble loss 以权重 1 加进 `total_loss`。

## 4. 消融设置

### 4.1 配置文件

新增 7 个配置，加上默认的 `atari_visual`，共 8 种开关组合（2³）。除表中列出的开关外，其余超参数都沿用 `atari_visual.py` 的默认值。

| `CONFIG` | 创新点 1 | 创新点 3 | 创新点 4 | 说明 |
|---|:-:|:-:|:-:|---|
| `atari_abl_none` | 关 | 关 | 关 | 原始 STORM（对照组） |
| `atari_abl_1` | 开 | 关 | 关 | 只开创新点 1 |
| `atari_abl_3` | 关 | 开 | 关 | 只开创新点 3 |
| `atari_abl_4` | 关 | 关 | 开 | 只开创新点 4 |
| `atari_abl_13` | 开 | 开 | 关 | 1 + 3 |
| `atari_abl_14` | 开 | 关 | 开 | 1 + 4 |
| `atari_abl_34` | 关 | 开 | 开 | 3 + 4 |
| `atari_visual` | 开 | 开 | 开 | 全开（默认） |

每个配置文件只有一行 `OVERRIDES`，例如 `configs/atari_abl_13.py`：

```python
from configs.atari_visual import Params, build_with

OVERRIDES = dict(num_prior_heads=0)


def build(env_name, seed):
    return build_with(env_name, seed, **OVERRIDES)
```

开关的含义：`scheduled_sampling_prob=0`、`wm_priority_fraction=0`、`num_prior_heads=0` 分别关闭创新点 1、3、4。**全关与原始 STORM 的代码路径一致**：不走第二次前向，用原 `sample`，不创建先验头。

### 4.2 训练与评估命令

`scripts/train.sh` 和 `scripts/eval.sh` 现在用环境变量选择游戏、种子、配置，`run_name` 自动生成为 `<游戏>-<配置>-seed<种子>`，不同配置和种子不会互相覆盖检查点（原来的 `<游戏>-DEV` 会覆盖）。

```bash
cd /root/autodl-tmp/OC-STORM-Sim

# 默认：Freeway，seed 46，全开
bash scripts/train.sh
bash scripts/eval.sh

# 指定游戏 / 种子 / 配置
ENV=Breakout SEED=1 CONFIG=atari_abl_none bash scripts/train.sh
ENV=Breakout SEED=1 CONFIG=atari_abl_none bash scripts/eval.sh    # 三个变量必须与训练时一致
```

一次完整消融（单个游戏、单个种子）：

```bash
for cfg in atari_abl_none atari_abl_1 atari_abl_3 atari_abl_4 atari_abl_13 atari_abl_14 atari_abl_34 atari_visual; do
    ENV=Breakout SEED=1 CONFIG=$cfg bash scripts/train.sh
    ENV=Breakout SEED=1 CONFIG=$cfg bash scripts/eval.sh | tee -a ablation_Breakout_seed1.log
done
```

### 4.3 如何读结果

- `eval.py` 结尾会打印 `Eval over N episodes: mean return = ..., std = ...`，同时把每局回报写进 `eval_results/<run_name>_episode_return.csv`。
- **单个种子的差异很容易被噪声淹没**。Atari-100k 通常要多个种子取平均（你之前的 `result.txt` 是 5 个种子）。建议至少 3 个种子再判断某个模块是否有效，`scripts/summarize_result.py` 的思路可以直接复用。
- 建议的对照顺序：先看 `atari_abl_none` vs `atari_abl_1/3/4`（单模块增益），再看两两组合与全开是否有叠加或抵消。

### 4.4 训练日志与汇总脚本

三个模块都新增了 TensorBoard 指标。训练结束后用汇总脚本把它们压成一段文本：

```bash
python scripts/summarize_run.py Freeway-atari_visual-seed46      # 给 run_name，自动取最新日志目录
# 或者给完整目录：python scripts/summarize_run.py runs/Freeway-atari_visual-seed46-<时间戳>-0
```

它把每个指标在整个训练过程中平均切成 10 段，输出各段均值（左边是训练初期，右边是末期），以及训练回合回报。**把这段输出贴给我就能按第 5 节的规则调参**。没有开启的模块对应的指标不会出现。

### 4.5 涉及的文件

修改的文件：`agents/visual/agent.py`、`agents/visual/world_model.py`、`agents/visual/modules/functions_losses.py`、`utils/replay_buffer.py`、`configs/atari_visual.py`、`scripts/train.sh`、`scripts/eval.sh`。

新增的文件：`configs/atari_abl_{none,1,3,4,13,14,34}.py`、`scripts/summarize_run.py`。

先验头是 `WorldModel` 的子模块，所以含先验头的检查点与不含先验头的**不能互相加载**。评估时必须用与训练时相同的 `CONFIG`。

## 5. 调参：跑一遍之后怎么改

这一节的判断规则都是我根据代码机制推出来的经验规则，**没有实验验证**，阈值只是起点。流程是：用某个配置跑完一遍 → `summarize_run.py` 输出 → 对照下表决定调哪个参数 → 改 `configs/atari_visual.py`（或对应的 `atari_abl_*.py` 里的 `OVERRIDES`）→ 再跑。

### 5.1 创新点 1：`scheduled_sampling_prob`

看 `ss_mix_ratio`、`ss_kl_teacher_forced`、`ss_kl_mixed`、`dynamics_real_kl_div`、`reconstruction_loss`。

| 观察到的现象 | 说明 | 怎么改 |
|---|---|---|
| `ss_mix_ratio` 不在 p 附近 | 实现有问题 | 先排查代码，不调参 |
| `ss_kl_mixed` 与 `ss_kl_teacher_forced` 几乎相等（比值接近 1） | 模型的一步预测已经足够好，混入的预测样本和真实样本相近，等于没起作用 | 提高 p（0.3 → 0.5） |
| `ss_kl_mixed` 比 `ss_kl_teacher_forced` 大很多（比如 1.5 倍以上），并且 `reconstruction_loss` / `reward_loss` 比对照组差 | 混入噪声太大，损害了对真实序列的拟合 | 降低 p（0.3 → 0.1~0.15） |
| 训练前期 WM loss 下降明显比对照组慢 | 模型还不会预测就被迫用自己的预测 | 引入 p 的 warm-up：前 N 步 p=0，之后再升到目标值（当前代码没有，需要加） |
| 与对照组相比，训练 loss 基本一致，最终得分无差异 | 可能是这个游戏对想象误差不敏感 | 不用动，换游戏看 |

补充：`ss_kl_*` 在训练初期两者都很大，意义不大，看后半段。

### 5.2 创新点 3：`wm_priority_fraction`、`wm_priority_init_loss`、`wm_priority_temperature`

看 `Replay/wm_loss_mean`、`Replay/wm_loss_max`、`Replay/untouched_ratio`、`Replay/eff_size`、`WorldModel/per_sample_loss_mean`。这四个 `Replay/*` 指标每 100 次更新记一次。

先说明几个量：

- `untouched_ratio` 是缓冲区里优先级仍等于初值的窗口占比，也就是从未被世界模型训练过的窗口。
- `eff_size = 1/Σp²` 是 softmax 分布的有效样本数。值越小，优先级路径越集中在少数窗口；它最大等于缓冲区窗口数（完全均匀）。

| 观察到的现象 | 说明 | 怎么改 |
|---|---|---|
| `wm_loss_mean` 远低于 `init_loss`（10），且 `untouched_ratio` 长期很高 | 没被抽到过的窗口始终比训练过的窗口优先级高，优先级路径主要在反复抽"没见过的窗口"，不是在抽"loss 高的窗口" | 把 `wm_priority_init_loss` 调到接近训练中后期的 `wm_loss_mean` 或 `wm_loss_max` 的量级 |
| `wm_loss_mean` 远高于 `init_loss` | 新数据优先级反而比旧数据低，新数据得不到优先 | 提高 `wm_priority_init_loss` 到 `wm_loss_max` 附近 |
| `eff_size` 很小（比如几十，而缓冲区有上万窗口） | 优先级分布过尖，反复训练少数窗口，容易过拟合 | 提高 `wm_priority_temperature`（1 → 2 → 4），或降低 `wm_priority_fraction` |
| `eff_size` 接近缓冲区窗口数 | softmax 太平，优先级几乎等于均匀采样，模块没起作用 | 降低 `wm_priority_temperature`（1 → 0.5），或提高 `wm_priority_fraction` |
| `wm_loss_max` 随训练持续飙升到很大 | 少数窗口（如含得分事件的窗口）loss 异常高，会垄断优先级 | 提高温度，或对 loss 做上限裁剪（当前代码没有） |

`Replay/*` 的数值依赖游戏和训练阶段，**我没有给出"应该是多少"的绝对值**，请以第一遍实际跑出来的数值为准，调的是 `init_loss` 与实际 loss 量级之间的关系。

### 5.3 创新点 4：`num_prior_heads`、`intrinsic_reward_weight`

看 `Imagine/intrinsic_reward_mean`、`Imagine/env_reward_abs_mean`、`WorldModel/prior_ensemble_jsd`、`WorldModel/prior_ensemble_loss`、`ActorCritic/S`、`ActorCritic/norm_ratio`、`ActorCritic/entropy_loss`。

关键比值是：`weight × intrinsic_reward_mean` 对 `env_reward_abs_mean`，即内在奖励相对环境奖励的大小。

| 观察到的现象 | 说明 | 怎么改 |
|---|---|---|
| 内在奖励贡献 `weight × intrinsic_reward_mean` 比 `env_reward_abs_mean` 小两个数量级以上 | 内在奖励基本被环境奖励淹没（分数大的游戏常见），模块等于没开 | 提高 `intrinsic_reward_weight`（1 → 10 → 100），目标是让两者在同一量级内 |
| 内在奖励贡献比 `env_reward_abs_mean` 大（比如 3 倍以上），且训练回合回报明显比对照组低、`entropy_loss` 偏高 | 内在奖励占主导，策略在追逐"不确定"而不是得分 | 降低 `intrinsic_reward_weight`（1 → 0.3 → 0.1） |
| `norm_ratio` 一直是 1（即 `S < 1`） | STORM 的回报归一化 `max(1, S)` 没有放大优势，小量级的内在奖励很难影响策略 | 同上，提高权重；也可以考虑把内在奖励先按环境奖励的尺度归一化（当前代码没有） |
| `prior_ensemble_jsd` 训练初期就趋近 0 并一直为 0 | 各头预测几乎相同，没有分歧信号。可能是头太少、容量太小或数据太少 | 增加 `num_prior_heads`（4 → 8）；检查 `prior_ensemble_loss` 是否在下降 |
| `prior_ensemble_loss` 不下降或震荡 | 头训练不动 | 当前头与主干共用 `lr=1e-4`；可以给头单独设更高学习率（当前代码没有），或降低 `loss` 的权重 |
| `prior_ensemble_loss` 的量级和 `reconstruction_loss` 相当甚至更大，且 `WorldModel/grad_norm` 比对照组明显变大 | 头的 loss 占梯度范数预算（裁剪阈值 1000）过多，可能影响主干更新 | 给 `prior_ensemble_loss` 乘一个小于 1 的权重（当前代码没有，需要加） |
| `intrinsic_reward_mean` 在想象中随步数明显上升 | 想象越往后越偏离真实数据，JSD 因分布外变大，不一定是真的"有价值的不确定" | 缩短 `imagine_batch_length`，或只在前几步加内在奖励（当前代码没有） |

### 5.4 通用检查

不管开哪个模块，对照 `atari_abl_none` 的这几项，如果新配置明显更差，应该先排查而不是继续调：

- `WorldModel/reconstruction_loss`、`reward_loss`、`total_loss` 的后期数值
- `WorldModel/grad_norm`（有没有异常大，相对裁剪阈值 1000）
- `sample/episode_return` 的趋势
- `ActorCritic/entropy_loss`（策略是否过早塌缩，或始终不收敛）

### 5.5 给我的数据清单

第一遍跑完后，请贴以下内容：

1. 所用命令（游戏、种子、`CONFIG`）；
2. `python scripts/summarize_run.py <run_name>` 的完整输出；
3. `eval.py` 末尾的 `mean return` 一行；
4. 如果有对照组（`atari_abl_none`），也贴它的同样两项。

只有一个配置的数据时，我只能按 5.2、5.3 的"数值关系"类规则（`init_loss` 与 `wm_loss_mean`、内在奖励与环境奖励的比值等）调参；"是否提升得分"必须有对照组才能判断。

## 6. 局限：在 Simulus 有效，在 STORM 上不一定有效

这三个模块的参数是照搬（创新点 1 来自 `innovation.md`，3 和 4 来自 Simulus）。Simulus 与 STORM 在 loss 定义、奖励处理、actor-critic 上有差异，这些数值在 STORM 上没有验证过。**以下"可能"的判断都是从两边代码差异推出来的，没有实验支持。**

### 6.1 创新点 1

1. **一步预测的偏差有限。** `z_t^predict` 由真实上下文预测，不是自回归展开的结果，想象阶段里"预测误差再喂给预测"的级联没有被训练到（见 1.3）。`scheduled_sampling_prob` 即使调大，也只能覆盖这种一步误差。
2. **离散随机潜变量的预测本身是随机的。** STORM 的 `z` 是 32×32 类别采样，预测样本与真实样本不同，既可能是"模型有误差"，也可能只是"这一步本来就有多种合理结果"。后者被当成"需要纠正的误差"来训练，这与连续确定性场景下的 scheduled sampling 不同。
3. **目标是真实后验，输入是自己的预测。** 在预测错误的位置，`h_t` 基于错误的 `z_t` 去拟合真实的 `z_{t+1}`，可能让模型学到"忽略 `z_t`"，削弱对输入的依赖。`p` 太大时更容易出现。
4. **额外一次前向的开销没有测量。**

### 6.2 创新点 3

1. **`wm_priority_init_loss`（默认 10）与 loss 尺度是否匹配。**
   - `softmax(loss)` 没有温度（`T=1` 时），优先级的尖锐程度完全取决于 loss 的绝对值。
   - 这里的窗口 loss 定义是"32 个潜变量的 KL 之和 + reward loss"，数值范围与 Simulus 的 token 交叉熵不同，初值 10 不一定落在合适的位置。
   - 如果训练中窗口 loss 普遍低于 10，**从未被抽到过的旧窗口（仍是 10）会比所有已训练过的窗口都更优先**，抽样会被"未访问过的窗口"主导；反过来，如果 loss 普遍高于 10，新数据反而得不到优先。
   - 需要看 `Replay/wm_loss_mean` 和 `Replay/untouched_ratio`（5.2）。
2. **`wm_priority_fraction`（默认 0.3）。** STORM 的 `batch_size=32`，`ceil(0.3×32)=10` 个优先级样本。Atari-100k 的缓冲区总共 1e5 步，被反复抽中的高 loss 窗口占比可能比 Simulus 的数据集更高，导致过拟合到少数窗口；建议在 0.1 / 0.3 / 0.5 之间对比。
3. **优先级粒度。** 这里按窗口起点存优先级，窗口之间重叠 31/32，邻近起点的优先级不联动更新，优先级可能比较"陈旧"。
4. **STORM 的训练比例没有校准。** STORM 的 `min_train_ratio=max_train_ratio=1.0`（每采样一步训练一次，batch 32×32），数据被重复训练的次数与 Simulus 的 epoch 设置不同，重放优先级的收益可能不同。

### 6.3 创新点 4

1. **`intrinsic_reward_weight`（默认 1.0）：奖励尺度与真实奖励不匹配。**
   - Simulus 的 Atari 训练环境配置（`config/env/atari.yaml` 的 train 部分）里 `clip_reward: True`，真实奖励在 {-1, 0, 1}，内在奖励与之同量级（test 部分为 False）。
   - STORM 的 `train.py` 直接把环境原始奖励存进缓冲区（我检查过 `train.py` 和 `envs/atari/build_env.py`，没有符号截断），奖励尺度随游戏差异很大。
   - 因此同样 1.0 的权重，在高分游戏里内在奖励相对可以忽略，在稀疏低分游戏里可能占主导。**权重很可能需要按游戏调整**，或者把内在奖励按奖励尺度归一化（当前没有实现）。
   - Simulus 在 DMC / BiGym 里用的是 10，说明这个数值本身就与环境相关。
2. **STORM 的 actor-critic 对回报做 `max(1, S)` 归一化**（`policy.py`，`S` 是 λ-return 的 5%~95% 分位差）。JSD 数值很小时，加入内在奖励后 `S` 仍可能小于 1，这时优势没有被放大，内在奖励的影响会很弱；反之奖励大的游戏里它又会被 `S` 压小。
3. **先验头的容量与训练量。** K 个头各自只用 batch 的 1/K（`batch_size=32, K=4`，每头 8 个窗口），数据量很少，头与头的差异可能主要来自训练噪声，而不是真正的认知不确定性。可调：`num_prior_heads`（2~8）、头的隐层宽度、是否给头单独的学习率（当前与主干共用 `lr=1e-4`）。
4. **`prior_ensemble_loss` 在 `total_loss` 里的权重（当前为 1）。** 它对 32 个潜变量求和，数值和重建项同量级；由于输入被 `detach`，它不会影响主干梯度，但会占用 `clip_grad_norm_(max=1000)` 的梯度范数预算与 Adam 的共享状态，权重没有调过。
5. **训练与想象的分布差异。** 头在"真实后验上下文得到的 `h_t`"上训练，在想象时的输入是"由模型自己采样的 z 得到的 `h_t`"。想象越往后偏离真实数据越远，JSD 会因分布外而上升，这既可能是有用的探索信号，也可能只是模型误差。`imagine_batch_length=16` 与折扣 `gamma=0.985` 会影响这种误差累积的影响。
6. **JSD 的聚合方式。** 这里对 32 个潜变量取平均，每个变量的 JSD 上界约为 `ln K`；Simulus 是 token 取平均、模态求和。如果改成"对 32 个变量求和"，数值会放大 32 倍，`intrinsic_reward_weight` 需要对应改。
7. **真实奖励没有缩放项。** Simulus 有 `real_reward_weight`，这里没有，所以不能单独把真实奖励的权重降下来，只能调内在奖励的权重。

### 6.4 模块之间的交互

- 创新点 1 + 4：先验头的输入 `h_t` 来自混合输入的前向，训练分布与想象时更接近，但也更噪。
- 创新点 3 + 4：被优先采样到的窗口也会进入先验头的训练分块（已经打乱，所以每个头拿到的混合比例相近），高 loss 窗口会让先验头更偏向这些样本，JSD 的数值分布会和只开创新点 4 时不同。
- 创新点 1 + 3：优先级 loss 来自混合输入的前向，所以窗口的 loss 会比纯 teacher forcing 的更高、更噪，`wm_priority_init_loss` 需要在开启创新点 1 之后重新对照 `Replay/wm_loss_mean` 校准。
