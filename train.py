"""
Implements training loop for the MDP agent from Duan et al., 2016
- 'RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning'
"""

from typing import List, Dict, Optional, Callable
from collections import deque
from functools import partial
import argparse

import torch as tc
import numpy as np
from mpi4py import MPI

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.envs.bandit_env import BanditEnv
from rl2.envs.mdp_env import MDPEnv

from rl2.agents.abstract import StatefulPolicyNet, StatefulValueNet
from rl2.agents.bandit_agent import PolicyNetworkMAB, ValueNetworkMAB
from rl2.agents.mdp_agent import PolicyNetworkMDP, ValueNetworkMDP

from rl2.utils.comm_util import get_comm, sync_state, sync_grads
from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.constants import ROOT_RANK


class MetaEpisode:
    def __init__(self, episode_len, num_episodes, dummy_obs):
        self.horizon = episode_len * num_episodes
        self.obs = np.array([dummy_obs for _ in range(self.horizon)])
        self.acs = np.zeros(self.horizon, 'int64')
        self.rews = np.zeros(self.horizon, 'float32')
        self.dones = np.zeros(self.horizon, 'float32')
        self.logpacs = np.zeros(self.horizon, 'float32')
        self.vpreds = np.zeros(self.horizon, 'float32')
        self.advs = np.zeros(self.horizon, 'float32')
        self.tdlam_rets = np.zeros(self.horizon, 'float32')


@tc.no_grad()
def generate_meta_episode(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        episode_len: int,
        num_episodes: int
    ) -> MetaEpisode:
    """
    Generates a meta-episode: a sequence of episodes concatenated together,
    with decisions being made by a recurrent agent with state preserved
    across episode boundaries.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network.
        episode_len: timesteps per episode.
        num_episodes: episodes per meta-episode.

    Returns:
        meta_episode: an instance of the meta-episode class.
    """

    env.new_env()
    t = 0
    o_t = env.reset()
    a_tm1 = np.array([0])
    r_tm1 = np.array([0.0])
    d_tm1 = np.array([1.0])
    h_tm1_policy_net = policy_net.initial_state(batch_size=1)
    h_tm1_value_net = value_net.initial_state(batch_size=1)

    meta_episode = MetaEpisode(
        episode_len=episode_len,
        num_episodes=num_episodes,
        dummy_obs=o_t)

    for episode_num in range(0, num_episodes):
        for episode_step in range(0, episode_len):
            pi_dist_t, h_t_policy_net = policy_net(
                curr_obs=tc.LongTensor(o_t),
                prev_action=tc.LongTensor(a_tm1),
                prev_reward=tc.FloatTensor(r_tm1),
                prev_done=tc.FloatTensor(d_tm1),
                prev_state=h_tm1_policy_net)

            vpred_t, h_t_value_net = value_net(
                curr_obs=tc.LongTensor(o_t),
                prev_action=tc.LongTensor(a_tm1),
                prev_reward=tc.FloatTensor(r_tm1),
                prev_done=tc.FloatTensor(d_tm1),
                prev_state=h_tm1_value_net)

            a_t = pi_dist_t.sample()
            log_prob_a_t = pi_dist_t.log_prob(a_t)

            o_tp1, r_t, done_t, _ = env.step(a_t, auto_reset=True)

            meta_episode.obs[t] = o_t
            meta_episode.acs[t] = a_t.squeeze(0).detach().numpy()
            meta_episode.rews[t] = r_t
            meta_episode.dones[t] = float(done_t)
            meta_episode.logpacs[t] = log_prob_a_t.squeeze(0).detach().numpy()
            meta_episode.vpreds[t] = vpred_t.squeeze(0).detach().numpy()

            o_t = o_tp1
            a_tm1 = np.array([meta_episode.acs[t]])
            r_tm1 = np.array([meta_episode.rews[t]])
            d_tm1 = np.array([meta_episode.dones[t]])
            h_tm1_policy_net = h_t_policy_net
            h_tm1_value_net = h_t_value_net
            t += 1

    return meta_episode


@tc.no_grad()
def assign_credit(
        meta_episode: MetaEpisode,
        gamma: float,
        lam: float
    ) -> MetaEpisode:
    """
    Compute td lambda returns and generalized advantage estimates.

    Note that in the meta-episodic setting of RL^2, the objective is
    to maximize the expected discounted return of the meta-episode,
    so we do not utilize the usual 'done' masking in this function.

    Args:
        meta_episode: meta-episode.
        gamma: discount factor.
        lam: GAE decay parameter.

    Returns:
        meta_episode: an instance of the meta-episode class,
        with generalized advantage estimates and td lambda returns computed.
    """
    T = len(meta_episode.acs)
    for t in reversed(range(0, T)):  # T-1, ..., 0.
        r_t = meta_episode.rews[t]
        V_t = meta_episode.vpreds[t]
        V_tp1 = meta_episode.vpreds[t+1] if t+1 < T else 0.0
        A_tp1 = meta_episode.advs[t+1] if t+1 < T else 0.0
        delta_t = -V_t + r_t + gamma * V_tp1
        A_t = delta_t + gamma * lam * A_tp1
        meta_episode.advs[t] = A_t

    meta_episode.tdlam_rets = meta_episode.vpreds + meta_episode.advs
    return meta_episode


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
    value_loss = tc.mean(tc.square(mb_tdlam_rets - vpreds_new))

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
        value_checkpoint_fn: Callable[[int], None]
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

    Returns:
        None
    """
    comm = get_comm()
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
            print(f"mean meta-episode return: {np.mean(meta_ep_returns)}")
            print("-" * 100)
            policy_checkpoint_fn(pol_iter)
            value_checkpoint_fn(pol_iter)


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script.""")
    parser.add_argument("--mode", choices=['mdp', 'bandit'], default='mdp')
    parser.add_argument("--max_pol_iters", type=int, default=1000)
    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=10)
    parser.add_argument("--episodes_per_meta_episode", type=int, default=10)
    parser.add_argument("--meta_episodes_per_policy_update", type=int, default=30000//100)
    parser.add_argument("--meta_episodes_per_actor_batch", type=int, default=60)
    parser.add_argument("--ppo_opt_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip_param", type=float, default=0.10)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.3)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-3)
    parser.add_argument("--experiment_seed", type=int, default=0) # not yet used
    return parser


def create_env(mode, num_states, num_actions, episode_len):
    if mode == 'mdp':
        env = MDPEnv(
            num_states=num_states,
            num_actions=num_actions,
            max_ep_length=episode_len)
        return env

    env = BanditEnv(num_actions=num_actions)
    return env


def create_agent(mode, num_states, num_actions):
    if mode == 'mdp':
        policy_net = PolicyNetworkMDP(
            num_states=num_states, num_actions=num_actions)
        value_net = ValueNetworkMDP(
            num_states=num_states, num_actions=num_actions)
        return policy_net, value_net

    policy_net = PolicyNetworkMAB(num_actions=num_actions)
    value_net = ValueNetworkMAB(num_actions=num_actions)
    return policy_net, value_net


def main():
    args = create_argparser().parse_args()

    # create env and learning system.
    env = create_env(
        mode=args.mode,
        num_states=args.num_states,
        num_actions=args.num_actions,
        episode_len=args.episode_len)

    policy_net, value_net = create_agent(
        mode=args.mode,
        num_states=args.num_states,
        num_actions=args.num_actions)

    policy_optimizer = tc.optim.Adam(
        params=policy_net.parameters(),
        lr=args.adam_lr,
        eps=args.adam_eps)
    value_optimizer = tc.optim.Adam(
        params=value_net.parameters(),
        lr=args.adam_lr,
        eps=args.adam_eps)

    policy_scheduler = None
    value_scheduler = None

    # load checkpoint, if applicable.
    comm = get_comm()
    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/policy_net",
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None)

        b = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/value_net",
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None)

        if a != b:
            raise RuntimeError(
                "Policy and value iterates not aligned in latest checkpoint!")
        pol_iters_so_far = a

    # sync state.
    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        comm=comm,
        root=ROOT_RANK)
    sync_state(
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
        comm=comm,
        root=ROOT_RANK)

    # make callback functions for checkpointing.
    policy_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/policy_net",
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler)

    value_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/value_net",
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler)

    # run it!
    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        episode_len=args.episode_len,
        episodes_per_meta_episode=args.episodes_per_meta_episode,
        meta_episodes_per_actor_batch=args.meta_episodes_per_actor_batch,
        meta_episodes_per_policy_update=args.meta_episodes_per_policy_update,
        ppo_opt_epochs=args.ppo_opt_epochs,
        ppo_clip_param=args.ppo_clip_param,
        ppo_ent_coef=args.ppo_ent_coef,
        discount_gamma=args.discount_gamma,
        gae_lambda=args.gae_lambda,
        max_pol_iters=args.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn)


if __name__ == '__main__':
    main()
