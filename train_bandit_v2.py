"""
Implements training loop for the bandit agent from Duan et al., 2016
- 'RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning'
"""

import argparse
from functools import partial

import torch as tc

from rl2.envs.bandit_env import BanditEnv
from rl2.agents_v2.preprocessing.tabular import MABPreprocessing
from rl2.agents_v2.architectures.gru import GRU
from rl2.agents_v2.heads.policy_heads import LinearPolicyHead
from rl2.agents_v2.heads.value_heads import LinearValueHead
from rl2.agents_v2.integration.policy_net import StatefulPolicyNet
from rl2.agents_v2.integration.value_net import StatefulValueNet
from rl2.algos_v2.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK
from rl2.utils.optim_util import get_weight_decay_param_groups


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2 bandit agent.""")
    parser.add_argument("--max_pol_iters", type=int, default=600)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--use_wn", type=int, choices=[0,1], default=0)
    parser.add_argument("--forget_bias", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--episodes_per_meta_episode", type=int, default=10)
    parser.add_argument("--meta_episodes_per_policy_update", type=int, default=30000//10)
    parser.add_argument("--meta_episodes_per_actor_batch", type=int, default=60)
    parser.add_argument("--ppo_opt_epochs", type=int, default=8)
    parser.add_argument("--ppo_clip_param", type=float, default=0.10)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.3)
    parser.add_argument("--standardize_advs", type=int, choices=[0,1], default=0)
    parser.add_argument("--adam_lr", type=float, default=2e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    parser.add_argument("--adam_wd", type=float, default=0.01)
    parser.add_argument("--experiment_seed", type=int, default=0) # not yet used
    return parser


def main():
    args = create_argparser().parse_args()
    comm = get_comm()

    # create env.
    env = BanditEnv(num_actions=args.num_actions)

    # create learning system.
    policy_net = StatefulPolicyNet(
        preprocessing=MABPreprocessing(num_actions=args.num_actions),
        architecture=GRU(
            input_dim=args.num_actions+2,
            hidden_dim=args.num_features,
            forget_bias=args.forget_bias,
            use_ln=True,
            reset_after=True),
        policy_head=LinearPolicyHead(
            num_actions=args.num_actions,
            num_features=args.num_features)
    )

    value_net = StatefulValueNet(
        preprocessing=MABPreprocessing(num_actions=args.num_actions),
        architecture=GRU(
            input_dim=args.num_actions+2,
            hidden_dim=args.num_features,
            forget_bias=args.forget_bias,
            use_ln=True,
            reset_after=True),
        value_head=LinearValueHead(
            num_features=args.num_features)
    )

    policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(policy_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps)
    value_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(value_net, args.adam_wd),
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
        episode_len=1,
        episodes_per_meta_episode=args.episodes_per_meta_episode,
        meta_episodes_per_actor_batch=args.meta_episodes_per_actor_batch,
        meta_episodes_per_policy_update=args.meta_episodes_per_policy_update,
        ppo_opt_epochs=args.ppo_opt_epochs,
        ppo_clip_param=args.ppo_clip_param,
        ppo_ent_coef=args.ppo_ent_coef,
        discount_gamma=args.discount_gamma,
        gae_lambda=args.gae_lambda,
        standardize_advs=bool(args.standardize_advs),
        max_pol_iters=args.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn,
        comm=comm)


if __name__ == '__main__':
    main()
