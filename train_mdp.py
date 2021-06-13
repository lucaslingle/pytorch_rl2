"""
Implements training loop for the bandit agent from Duan et al., 2016
- 'RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning'
"""

import argparse
import os
import pickle
from functools import partial

import torch as tc

from rl2.agents.mdp_agent import PolicyNetworkMDP, ValueNetworkMDP
from rl2.envs.mdp_env import MDPEnv
from rl2.algos.ppo import training_loop

from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.constants import ROOT_RANK


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2 MDP agent.""")
    parser.add_argument("--max_pol_iters", type=int, default=6000)
    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=10)
    parser.add_argument("--episodes_per_meta_episode", type=int, default=10)
    parser.add_argument("--meta_episodes_per_policy_update", type=int, default=30000//100)
    parser.add_argument("--meta_episodes_per_actor_batch", type=int, default=60)
    parser.add_argument("--ppo_opt_epochs", type=int, default=40)
    parser.add_argument("--ppo_clip_param", type=float, default=0.10)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.3)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-3)
    parser.add_argument("--experiment_seed", type=int, default=0) # not yet used
    return parser


def create_env(num_states, num_actions, episode_len, checkpoint_dir, comm):
    env = MDPEnv(
        num_states=num_states,
        num_actions=num_actions,
        max_episode_length=episode_len)

    reward_means = env.reward_means
    dirichlet_conc_params = env.dirichlet_conc_params

    # on process with rank zero...
    if comm.Get_rank() == ROOT_RANK:
        env_base_path = os.path.join(checkpoint_dir, 'mdp_env')
        os.makedirs(env_base_path, exist_ok=True)

        # deserialize reward means from a pickled file if present,
        # else make it
        fp1 = os.path.join(env_base_path, 'reward_means')
        if os.path.exists(fp1):
            with open(fp1, 'rb') as f1:
                reward_means = pickle.load(f1)
        else:
            with open(fp1, 'wb') as f1:
                pickle.dump(reward_means, f1, protocol=pickle.HIGHEST_PROTOCOL)

        # deserialize dirichlet conc params from a pickled file if present,
        # else make it
        fp2 = os.path.join(env_base_path, 'dirichlet_conc_params')
        if os.path.exists(fp2):
            with open(fp2, 'rb') as f2:
                dirichlet_conc_params = pickle.load(f2)
        else:
            with open(fp2, 'wb') as f2:
                pickle.dump(
                    dirichlet_conc_params, f2, protocol=pickle.HIGHEST_PROTOCOL)

    # now we broadcast contents stored in the reference variable on process zero
    # to all the other processes
    reward_means = comm.Bcast(reward_means, root=ROOT_RANK)
    dirichlet_conc_params = comm.Bcast(dirichlet_conc_params, root=ROOT_RANK)

    # ... and load them into the env.
    env.set_reward_means(reward_means)
    env.set_dirichlet_conc_params(dirichlet_conc_params)

    return env


def main():
    args = create_argparser().parse_args()
    comm = get_comm()

    # create env.
    env = create_env(
        num_states=args.num_states,
        num_actions=args.num_actions,
        episode_len=args.episode_len,
        checkpoint_dir=args.checkpoint_dir,
        comm=comm)

    # create learning system.
    policy_net = PolicyNetworkMDP(
        num_states=args.num_states,
        num_actions=args.num_actions)
    value_net = ValueNetworkMDP(
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
        value_checkpoint_fn=value_checkpoint_fn,
        comm=comm)


if __name__ == '__main__':
    main()
