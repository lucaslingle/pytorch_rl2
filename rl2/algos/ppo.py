"""
Implements ppo loss computations for training
stateful meta-reinforcement learning agents.
"""

from typing import List, Dict, Optional, Callable
from collections import deque

import torch as tc
import numpy as np
from mpi4py import MPI

from rl2.agents.abstract import StatefulPolicyNet, StatefulValueNet
from rl2.envs.abstract import MetaEpisodicEnv
from rl2.algos.common import (
    MetaEpisode,
    generate_meta_episode,
    assign_credit,
    huber_func,
)
from rl2.utils.comm_util import sync_grads
from rl2.utils.constants import ROOT_RANK


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
        meta_episodes: list of meta-episodes.
        policy_net: policy network.
        value_net: value network.
        clip_param: clip parameter for PPO.
        ent_coef: entropy coefficient for PPO.

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

    mb_obs = get_tensor('obs', 'long')
    mb_acs = get_tensor('acs', 'long')
    mb_rews = get_tensor('rews')
    mb_dones = get_tensor('dones')
    mb_logpacs = get_tensor('logpacs')
    mb_advs = get_tensor('advs')
    mb_tdlam_rets = get_tensor('tdlam_rets')

    # we are going to manually loop over the timesteps here,
    # for backprop thru time.
    B = len(meta_episodes)
    T = len(meta_episodes[0].acs)

    ac_dummy = tc.zeros(dtype=tc.int64, size=(B,))
    rew_dummy = tc.zeros(dtype=tc.float32, size=(B,))
    done_dummy = tc.ones(dtype=tc.float32, size=(B,))
    h_tm1_policy_net = policy_net.initial_state(batch_size=B)  # will be ref var
    h_tm1_value_net = value_net.initial_state(batch_size=B)    # will be ref var

    entropies = []
    logpacs_new = []
    vpreds_new = []
    for t in range(0, T):
        o_t = mb_obs[:, t]
        a_tm1 = ac_dummy if t == 0 else mb_acs[:, t-1]
        r_tm1 = rew_dummy if t == 0 else mb_rews[:, t-1]
        d_tm1 = done_dummy if t == 0 else mb_dones[:, t-1]

        pi_dist_t, h_t_policy_net = policy_net(
            curr_obs=o_t,
            prev_action=a_tm1,
            prev_reward=r_tm1,
            prev_done=d_tm1,
            prev_state=h_tm1_policy_net)

        vpred_t, h_t_value_net = value_net(
            curr_obs=o_t,
            prev_action=a_tm1,
            prev_reward=r_tm1,
            prev_done=d_tm1,
            prev_state=h_tm1_value_net)

        ent_t = pi_dist_t.entropy()
        entropies.append(ent_t)

        a_t = mb_acs[:, t]
        logprob_a_t_new = pi_dist_t.log_prob(a_t)
        logpacs_new.append(logprob_a_t_new)

        vpreds_new.append(vpred_t)

        h_tm1_policy_net = h_t_policy_net
        h_tm1_value_net = h_t_value_net

    # assemble relevant gradient-tracked quantities from our loop above.
    entropies = tc.stack(entropies, dim=1)
    logpacs_new = tc.stack(logpacs_new, dim=1)
    vpreds_new = tc.stack(vpreds_new, dim=1)

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
    #value_loss = tc.mean(tc.square(mb_tdlam_rets - vpreds_new))
    value_loss = tc.mean(huber_func(mb_tdlam_rets, vpreds_new))

    # clipfrac
    clipfrac = tc.mean(tc.greater(surr1, surr2).float())

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "meanent": meanent,
        "clipfrac": clipfrac
    }


def training_loop(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        policy_optimizer: tc.optim.Optimizer,
        value_optimizer: tc.optim.Optimizer,
        policy_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        value_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        episode_len: int,
        episodes_per_meta_episode: int,
        meta_episodes_per_actor_batch: int,
        meta_episodes_per_policy_update: int,
        ppo_opt_epochs: int,
        ppo_clip_param: float,
        ppo_ent_coef: float,
        discount_gamma: float,
        gae_lambda: float,
        max_pol_iters: int,
        pol_iters_so_far: int,
        policy_checkpoint_fn: Callable[[int], None],
        value_checkpoint_fn: Callable[[int], None],
        comm: type(MPI.COMM_WORLD),
    ) -> None:
    """
    Train a stateful RL^2 agent via PPO to maximize discounted cumulative reward
    in Tabular MDPs, sampled from the distribution used in Duan et al., 2016.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network,
        policy_optimizer: policy optimizer.
        value_optimizer: value optimizer.
        policy_scheduler: policy lr scheduler.
        value_scheduler: value lr scheduler.
        episode_len: timesteps per episode.
        episodes_per_meta_episode: episodes per meta-episode.
        meta_episodes_per_actor_batch: meta-episodes per batch on each process.
        meta_episodes_per_policy_update: meta-episodes per policy improvement
            on each process.
        ppo_opt_epochs: optimization epochs for proximal policy optimization.
        ppo_clip_param: clip parameter for proximal policy optimization.
        ppo_ent_coef: entropy bonus coefficient for proximal policy optimization
        discount_gamma: discount factor gamma.
        gae_lambda: decay parameter lambda for generalized advantage estimation.
        max_pol_iters: the maximum number policy improvements to make.
        pol_iters_so_far: the number of policy improvements made so far.
        policy_checkpoint_fn: a callback for saving checkpoints of policy net.
        value_checkpoint_fn: a callback for saving checkpoints of value net.
        comm: mpi comm_world communicator object.

    Returns:
        None
    """
    meta_ep_returns = deque(maxlen=1000)

    for pol_iter in range(pol_iters_so_far, max_pol_iters):

        # collect meta-episodes...
        meta_episodes = list()
        for _ in range(0, meta_episodes_per_policy_update):
            # collect one meta-episode and append it to the list
            meta_episode = generate_meta_episode(
                env=env,
                policy_net=policy_net,
                value_net=value_net,
                episode_len=episode_len,
                num_episodes=episodes_per_meta_episode)
            meta_episode = assign_credit(
                meta_episode=meta_episode,
                gamma=discount_gamma,
                lam=gae_lambda)
            meta_episodes.append(meta_episode)

            # logging
            l_meta_ep_returns = [np.sum(meta_episode.rews)]
            g_meta_ep_returns = comm.allgather(l_meta_ep_returns)
            g_meta_ep_returns = [x for loc in g_meta_ep_returns for x in loc]
            meta_ep_returns.extend(g_meta_ep_returns)

        # update policy...
        for opt_epoch in range(ppo_opt_epochs):
            idxs = np.random.permutation(meta_episodes_per_policy_update)
            for i in range(0, meta_episodes_per_policy_update, meta_episodes_per_actor_batch):
                mb_idxs = idxs[i:i+meta_episodes_per_actor_batch]
                mb_meta_eps = [meta_episodes[idx] for idx in mb_idxs]
                losses = compute_losses(
                    meta_episodes=mb_meta_eps,
                    policy_net=policy_net,
                    value_net=value_net,
                    clip_param=ppo_clip_param,
                    ent_coef=ppo_ent_coef)

                policy_optimizer.zero_grad()
                losses['policy_loss'].backward()
                sync_grads(model=policy_net, comm=comm)
                policy_optimizer.step()
                if policy_scheduler:
                    policy_scheduler.step()

                value_optimizer.zero_grad()
                losses['value_loss'].backward()
                sync_grads(model=value_net, comm=comm)
                value_optimizer.step()
                if value_scheduler:
                    value_scheduler.step()

            # logging
            global_losses = {}
            for name in losses:
                loss_sum = comm.allreduce(losses[name], op=MPI.SUM)
                loss_avg = loss_sum / comm.Get_size()
                global_losses[name] = loss_avg

            if comm.Get_rank() == ROOT_RANK:
                print(f"pol update {pol_iter}, opt_epoch: {opt_epoch}...")
                for name, value in global_losses.items():
                    print(f"\t{name}: {value:>0.6f}")

        # misc.: print metrics, save checkpoint.
        if comm.Get_rank() == ROOT_RANK:
            print("-" * 100)
            print(f"mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            print("-" * 100)
            policy_checkpoint_fn(pol_iter)
            value_checkpoint_fn(pol_iter)
