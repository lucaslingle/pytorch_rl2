"""
Implements ppo loss computations for training
stateful meta-reinforcement learning agents.
"""

from typing import List, Dict, Optional, Any
from collections import deque

import torch as tc
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.algos.common import (
    MetaEpisode, generate_meta_episode, assign_credit, huber_func
)
from rl2.utils.checkpoint_util import save_checkpoints
from rl2.utils.typing_util import (
    StatefulPolicyNet,
    StatefulValueNet,
    Optimizer,
    Scheduler,
)


def compute_losses(
        meta_episodes: List[MetaEpisode],
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        clip_param: float,
        ent_coef: float
    ) -> Dict[str, tc.Tensor]:
    """
    Computes the losses for Proximal Policy Optimization.

    Args:
        meta_episodes: List of meta-episodes.
        policy_net: Policy network.
        value_net: Value network.
        clip_param: PPO clip parameter.
        ent_coef: PPO entropy bonus coefficient.

    Returns:
        loss_dict: a dictionary of losses.
    """
    def get_tensor(field, dtype=None):
        mb_field = np.stack(
            list(map(lambda metaep: getattr(metaep, field), meta_episodes)),
            axis=0)
        if dtype == 'long':
            return tc.LongTensor(mb_field)
        return tc.FloatTensor(mb_field)

    # minibatch data tensors
    mb_obs = get_tensor('obs', 'long')
    mb_acs = get_tensor('acs', 'long')
    mb_rews = get_tensor('rews')
    mb_dones = get_tensor('dones')
    mb_logpacs = get_tensor('logpacs')
    mb_advs = get_tensor('advs')
    mb_tdlam_rets = get_tensor('tdlam_rets')

    # input for loss calculations
    B = len(meta_episodes)
    ac_dummy = tc.zeros(dtype=tc.int64, size=(B,))
    rew_dummy = tc.zeros(dtype=tc.float32, size=(B,))
    done_dummy = tc.ones(dtype=tc.float32, size=(B,))

    curr_obs = mb_obs
    prev_action = tc.cat((ac_dummy.unsqueeze(1), mb_acs[:, 0:-1]), dim=1)
    prev_reward = tc.cat((rew_dummy.unsqueeze(1), mb_rews[:, 0:-1]), dim=1)
    prev_done = tc.cat((done_dummy.unsqueeze(1), mb_dones[:, 0:-1]), dim=1)
    prev_state_policy_net = policy_net.initial_state(batch_size=B)
    prev_state_value_net = value_net.initial_state(batch_size=B)

    # forward pass implements unroll for recurrent/attentive architectures.
    pi_dists, _ = policy_net(
        curr_obs=curr_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_done=prev_done,
        prev_state=prev_state_policy_net)

    vpreds, _ = value_net(
        curr_obs=curr_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_done=prev_done,
        prev_state=prev_state_value_net)

    entropies = pi_dists.entropy()
    logpacs_new = pi_dists.log_prob(mb_acs)
    vpreds_new = vpreds

    # entropy bonus
    meanent = tc.mean(entropies)
    policy_entropy_bonus = ent_coef * meanent

    # policy surrogate objective
    policy_ratios = tc.exp(logpacs_new - mb_logpacs)
    clipped_policy_ratios = tc.clip(policy_ratios, 1-clip_param, 1+clip_param)
    surr1 = mb_advs * policy_ratios
    surr2 = mb_advs * clipped_policy_ratios
    policy_surrogate_objective = tc.mean(tc.min(surr1, surr2))

    # composite policy loss
    policy_loss = -(policy_surrogate_objective + policy_entropy_bonus)

    # value loss
    value_loss = tc.mean(huber_func(mb_tdlam_rets, vpreds_new))

    # clipfrac
    clipfrac = tc.mean(tc.greater(surr1, surr2).float())

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "meanent": meanent,
        "clipfrac": clipfrac
    }


def global_mean(metric, world_size):
    # for logging purposes only!
    global_metric = metric.detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.float().item() / world_size


def global_means(metrics, world_size):
    # for logging purposes only!
    return {k: global_mean(v, world_size) for k, v in metrics.items()}


def training_loop(
        rank: int,
        world_size: int,
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        policy_optimizer: Optimizer,
        policy_scheduler: Optional[Scheduler],
        value_net: StatefulValueNet,
        value_optimizer: Optimizer,
        value_scheduler: Optional[Scheduler],
        meta_episodes_per_policy_update: int,
        meta_episodes_per_learner_batch: int,
        meta_episode_len: int,
        ppo_opt_epochs: int,
        ppo_clip_param: float,
        ppo_ent_coef: float,
        discount_gamma: float,
        gae_lambda: float,
        pol_iters_so_far: int,
        max_pol_iters: int,
        checkpoint_dir: str,
        **kwargs: Dict[str, Any]
    ) -> None:
    """
    Training loop.

    Args:
        rank: Process rank.
        world_size: World size.
        env: MetaEpisodicEnv instance.
        policy_net: Policy network.
        policy_optimizer: Policy optimizer.
        policy_scheduler: Optional policy lr scheduler.
        value_net: Value network,
        value_optimizer: Value optimizer.
        value_scheduler: Optional value lr scheduler.
        meta_episodes_per_policy_update: Local meta-episodes per policy update.
        meta_episodes_per_learner_batch: Local meta-episodes per batch.
        meta_episode_len: Timesteps per meta-episode.
        ppo_opt_epochs: PPO optimization epochs.
        ppo_clip_param: PPO clip parameter.
        ppo_ent_coef: PPO entropy bonus coefficient.
        discount_gamma: Discount factor gamma.
        gae_lambda: GAE(lambda) decay parameter.
        pol_iters_so_far: Number of policy improvements made so far.
        max_pol_iters: Maximum number policy improvements to make.
        checkpoint_dir: Checkpoint directory for saving checkpoints.
        kwargs: Keyword args.

    Returns:
        None
    """
    meta_ep_returns = deque(maxlen=1000)
    for pol_iter in range(pol_iters_so_far, max_pol_iters):
        meta_episodes = list()
        for _ in range(0, meta_episodes_per_policy_update):
            meta_episode = generate_meta_episode(
                env=env,
                policy_net=policy_net,
                value_net=value_net,
                meta_episode_len=meta_episode_len)
            meta_episode = assign_credit(
                meta_episode=meta_episode,
                gamma=discount_gamma,
                lam=gae_lambda)
            meta_episodes.append(meta_episode)
            meta_ep_returns.append(np.sum(meta_episode.rews))

        for opt_epoch in range(ppo_opt_epochs):
            idxs = np.random.permutation(meta_episodes_per_policy_update)
            for i in range(0, meta_episodes_per_policy_update, meta_episodes_per_learner_batch):
                mb_idxs = idxs[i:i+meta_episodes_per_learner_batch]
                mb_meta_eps = [meta_episodes[idx] for idx in mb_idxs]
                losses = compute_losses(
                    meta_episodes=mb_meta_eps,
                    policy_net=policy_net,
                    value_net=value_net,
                    clip_param=ppo_clip_param,
                    ent_coef=ppo_ent_coef)
                policy_loss = losses.get('policy_loss')
                value_loss = losses.get('value_loss')

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                if policy_scheduler:
                    policy_scheduler.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                if value_scheduler:
                    value_scheduler.step()

            # logging
            global_losses = global_means(losses, world_size)
            if rank == 0:
                print(f"pol update {pol_iter}, opt_epoch: {opt_epoch}...")
                for name, value in global_losses.items():
                    print(f"\t{name}: {value:>0.6f}")

        # misc.: print metrics, save checkpoint.
        if rank == 0:
            print("-" * 100)
            print(f"mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            print("-" * 100)
            save_checkpoints(
                checkpoint_dir=checkpoint_dir,
                checkpointables={
                    'policy_net': policy_net,
                    'policy_optimizer': policy_optimizer,
                    'policy_scheduler': policy_scheduler
                },
                steps=pol_iter+1)
            save_checkpoints(
                checkpoint_dir=checkpoint_dir,
                checkpointables={
                    'value_net': value_net,
                    'value_optimizer': value_optimizer,
                    'value_scheduler': value_scheduler
                },
                steps=pol_iter+1)
