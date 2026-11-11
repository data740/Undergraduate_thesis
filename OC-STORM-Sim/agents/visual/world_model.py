import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Categorical
import einops
import einops.layers
import einops.layers.torch
from torch.cuda.amp import autocast

from .modules.functions_losses import SymLogTwoHotLoss
from .modules.attention_blocks import (
    get_causal_mask_with_batch_length,
    get_causal_mask,
    AttentionBlockKVCache,
    PositionwiseFeedForward,
)
from .modules.transformer_model import StochasticTransformerKVCache
from .policy import ActorCritic


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels, out_channels=stem_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
        )
        feature_width = 64 // 2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.SiLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels, out_channels=channels * 2, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.SiLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = einops.rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = einops.rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels * final_feature_width * final_feature_width, bias=False))
        backbone.append(
            einops.layers.torch.Rearrange("B L (C H W) -> (B L) C H W", C=last_channels, H=final_feature_width)
        )
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.SiLU(inplace=True))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels, out_channels=channels // 2, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.SiLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels, out_channels=original_in_channels, kernel_size=4, stride=2, padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = einops.rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    """
    Dist: abbreviation of distribution
    """

    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim * stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim * stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1 - mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = einops.rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = einops.rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class EnsemblePriorHead(nn.Module):
    """
    Simulus-style ensemble of K prior heads. Every head reads the (detached) transformer feature h_t and predicts
    the categorical distribution of the next posterior z_{t+1}. Gradients never reach the world-model trunk, and each
    head is trained on its own 1/K share of the batch, so the heads only disagree where the data is thin.
    The Jensen-Shannon divergence between the K predictions is the epistemic uncertainty used as intrinsic reward.
    """

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
        """
        feat: [B, L, H] hidden state h_t, target_logits: [B, L, stoch_dim, stoch_dim] posterior of z_{t+1}.
        The batch is split into K disjoint chunks, head k only sees chunk k.
        """
        feat = feat.detach()
        target_probs = F.softmax(target_logits.detach().float(), dim=-1)
        chunks = torch.tensor_split(torch.arange(feat.shape[0], device=feat.device), self.ensemble_size)
        total = feat.new_zeros((), dtype=torch.float32)
        for head, idx in zip(self.heads, chunks):
            log_prob = F.log_softmax(self._logits(head, feat[idx]).float(), dim=-1)
            ce = -(target_probs[idx] * log_prob).sum(dim=-1)  # [b, L, stoch_dim]
            total = total + ce.sum(dim=-1).mean()  # sum over latents like the KL terms, mean over batch / time
        return total

    @torch.no_grad()
    def estimate_uncertainty(self, feat):
        """JSD between the K heads, averaged over the latent variables -> [B, L]"""
        log_p = F.log_softmax(self.forward_all(feat).float(), dim=-1)  # [K, B, L, S, C]
        p = log_p.exp()
        mean_p = p.mean(dim=0)
        entropy_of_mean = -(mean_p * torch.log(mean_p.clamp_min(1e-8))).sum(dim=-1)
        mean_of_entropy = -(p * log_p).sum(dim=-1).mean(dim=0)
        jsd = (entropy_of_mean - mean_of_entropy).clamp_min(0.0)
        return jsd.mean(dim=-1)


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs) ** 2
        loss = einops.reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = einops.reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(
        self,
        input_channels,
        action_dims,
        num_objects,
        latent_width,
        transformer_max_length,
        transformer_hidden_dim,
        transformer_num_layers,
        transformer_num_heads,
        num_prior_heads=0,
        intrinsic_reward_weight=1.0,
        scheduled_sampling_prob=0.0,
    ):
        super().__init__()
        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.use_amp = True
        self.amp_tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.action_dims = action_dims
        self.num_objects = num_objects
        self.transformer_hidden_dim = transformer_hidden_dim
        self.latent_width = latent_width
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.encoder = EncoderBN(in_channels=input_channels, stem_channels=32, final_feature_width=4)
        self.state_decoder = DecoderBN(
            stoch_dim=1024,
            last_channels=self.encoder.last_channels,
            original_in_channels=input_channels,
            stem_channels=32,
            final_feature_width=4,
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=1024,
            action_dims=action_dims,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
        )

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels * 4 * 4,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=32,
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255, embedding_size=1024, transformer_hidden_dim=transformer_hidden_dim
        )
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
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode_obs(self, obs):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_hidden(self, latent, action):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            # Run the transformer over the context sequence and keep only the
            # final hidden state for the last time step.
            temporal_mask = get_causal_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_hidden = dist_feat[:, -1:]
        return last_hidden

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.straight_through_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def straight_through_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return einops.rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        """
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        """
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length + 1, 1024)
            hidden_size = (imagine_batch_size, imagine_batch_length + 1, self.transformer_hidden_dim)
            action_size = (imagine_batch_size, imagine_batch_length, len(self.action_dims))
            scalar_size = (imagine_batch_size, imagine_batch_length)

            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=torch.int32, device="cuda")
            self.log_prob_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=torch.int32, device="cuda")

    def imagine_data(
        self,
        agent: ActorCritic,
        sample_state,
        sample_action,
        imagine_batch_size,
        imagine_batch_length,
        log_video,
        logger,
    ):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.amp_tensor_dtype)
        obs_hat_list = []
        intrinsic_reward_sum = 0.0
        real_reward_abs_sum = 0.0

        context_length = sample_state.shape[1]
        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.amp_tensor_dtype)
        # context
        context_latent = self.encode_obs(sample_state)
        for i in range(context_length):
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i : i + 1], sample_action[:, i : i + 1], log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action, log_prob = agent.sample_with_log_prob(
                torch.cat([self.latent_buffer[:, i : i + 1], self.hidden_buffer[:, i : i + 1]], dim=-1),
                # self.hidden_buffer[:, i:i+1],
                temperature=1.0,
            )
            self.action_buffer[:, i : i + 1] = action
            self.log_prob_buffer[:, i : i + 1] = log_prob

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i : i + 1], self.action_buffer[:, i : i + 1], log_video=log_video
            )

            if self.prior_ensemble is not None:
                # disagreement of the K heads about z_{t+1} given h_t is a bonus for visiting uncertain transitions
                with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
                    intrinsic_reward = self.prior_ensemble.estimate_uncertainty(last_dist_feat)  # [B, 1], float32
                intrinsic_reward_sum = intrinsic_reward_sum + intrinsic_reward.mean()
                real_reward_abs_sum = real_reward_abs_sum + last_reward_hat.abs().float().mean()
                last_reward_hat = last_reward_hat + self.intrinsic_reward_weight * intrinsic_reward

            self.latent_buffer[:, i + 1 : i + 2] = last_latent
            self.hidden_buffer[:, i + 1 : i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i : i + 1] = last_reward_hat
            self.termination_hat_buffer[:, i : i + 1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[:: imagine_batch_size // 16])  # uniform sample vec_env

        if self.prior_ensemble is not None:
            self.last_intrinsic_reward_mean = float(intrinsic_reward_sum) / imagine_batch_length
            self.last_real_reward_abs_mean = float(real_reward_abs_sum) / imagine_batch_length

        if log_video:
            logger.log(
                "Imagine/predict_video_with_mask",
                torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy(),
            )

        return (
            torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1),
            self.action_buffer,
            self.log_prob_buffer,
            self.reward_hat_buffer,
            self.termination_hat_buffer,
        )

    def update(self, state, action, reward, termination, logger=None, return_per_sample_loss=False):
        self.train()
        batch_size, batch_length = state.shape[:2]

        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(state)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            state_hat = self.state_decoder(flattened_sample)

            # spatial-temporal transformer
            temporal_mask = get_causal_mask_with_batch_length(batch_length, flattened_sample.device)
            transformer_input = flattened_sample
            if self.scheduled_sampling_prob > 0:
                # scheduled sampling: pass 1 is teacher forced (no grad) and yields z_t^predict ~ prior(h_{t-1});
                # pass 2 feeds a mix of the posterior z_t and z_t^predict, all losses below come from pass 2
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
                    transformer_input = torch.cat([flattened_sample[:, :1], mixed_tail], dim=1)  # z_0 is always real
            dist_feat = self.storm_transformer(transformer_input, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func(state_hat, state)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(
                post_logits[:, 1:].detach(), prior_logits[:, :-1]
            )
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(
                post_logits[:, 1:], prior_logits[:, :-1].detach()
            )
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )
            if self.prior_ensemble is not None:
                # h_t -> z_{t+1}, input is detached inside, so the trunk is untouched
                prior_ensemble_loss = self.prior_ensemble.loss(dist_feat[:, :-1], post_logits[:, 1:])
                total_loss = total_loss + prior_ensemble_loss

            per_sample_loss = None
            if return_per_sample_loss:
                # priority of a window = next-latent KL (no free bits) + reward loss, averaged over time
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    # reward MSE loss, distinguish zero & non-zero situation
                    reward_hat_value = self.symlog_twohot_loss_func.decode(reward_hat)  # [B, L]
                    reward_hat_value, reward = reward_hat_value.flatten(), reward.flatten()
                    zero_reward_idices = reward == 0
                    if sum(zero_reward_idices) > 0:
                        reward_zero_mse_loss = (reward_hat_value[zero_reward_idices] - reward[zero_reward_idices]) ** 2
                        reward_zero_mse_loss = reward_zero_mse_loss.mean()
                        logger.log("WorldModel/reward_zero_mse_loss", reward_zero_mse_loss.item())
                    else:
                        logger.log("WorldModel/reward_zero_mse_loss", None)
                    if sum(zero_reward_idices) < len(reward):
                        reward_nonzero_mse_loss = (
                            reward_hat_value[~zero_reward_idices] - reward[~zero_reward_idices]
                        ) ** 2
                        reward_nonzero_mse_loss = reward_nonzero_mse_loss.mean()
                        logger.log("WorldModel/reward_nonzero_mse_loss", reward_nonzero_mse_loss.item())
                    else:
                        logger.log("WorldModel/reward_nonzero_mse_loss", None)

                    # termination F1 score
                    termination_hat_value = termination_hat > 0  # [B, L]
                    termination_hat_value, termination = termination_hat_value.flatten(), termination.flatten().bool()
                    tp = (termination_hat_value & termination).sum().float()
                    fp = (termination_hat_value & ~termination).sum().float()
                    tn = (~termination_hat_value & ~termination).sum().float()
                    fn = (~termination_hat_value & termination).sum().float()
                    if tp + fn > 0:
                        positive_recall = tp / (tp + fn)
                        logger.log("WorldModel/termination_positive_recall", positive_recall.item())
                    else:
                        logger.log("WorldModel/termination_positive_recall", None)
                    if fp + tn > 0:
                        negative_recall = tn / (fp + tn)
                        logger.log("WorldModel/termination_negative_recall", negative_recall.item())
                    else:
                        logger.log("WorldModel/termination_negative_recall", None)

            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            logger.log("WorldModel/grad_norm", float(grad_norm))
            if self.prior_ensemble is not None:
                logger.log("WorldModel/prior_ensemble_loss", prior_ensemble_loss.item())
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp
                ):
                    jsd = self.prior_ensemble.estimate_uncertainty(dist_feat[:, :-1].detach())
                logger.log("WorldModel/prior_ensemble_jsd", jsd.mean().item())
            if per_sample_loss is not None:
                logger.log("WorldModel/per_sample_loss_mean", per_sample_loss.mean().item())
            if self.scheduled_sampling_prob > 0:
                with torch.no_grad():
                    # real KL(post z_{t+1} || prior) without free bits: teacher forced input vs mixed input
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

        return per_sample_loss
