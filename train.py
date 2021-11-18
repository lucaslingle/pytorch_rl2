"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from functools import partial

import torch as tc

from rl2.envs.bandit_env import BanditEnv
from rl2.envs.mdp_env import MDPEnv

from rl2.agents.preprocessing.tabular import MABPreprocessing, MDPPreprocessing
from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.snail import SNAIL
from rl2.agents.architectures.transformer import TransformerXLI, SparseTransformerXLI
from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.algos.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK
from rl2.utils.optim_util import get_weight_decay_param_groups


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument("--environment", choices=['bandit', 'mdp'],
                        default='bandit')
    parser.add_argument("--num_states", type=int, default=10,
                        help="Ignored if environment is bandit.")
    parser.add_argument("--num_actions", type=int, default=5)

    ### Architecture
    parser.add_argument(
        "--architecture",
        choices=['gru', 'snail', 'trxli', 'sparse_transformer'],
        default='gru')
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Ignored if architecture is not gru/lstm.")

    ### Checkpointing
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')

    ### Training
    parser.add_argument("--max_pol_iters", type=int, default=12000)
    parser.add_argument("--episode_len", type=int, default=10,
                        help="Ignored if environment is bandit.")
    parser.add_argument("--episodes_per_meta_episode", type=int, default=10)
    parser.add_argument("--meta_episodes_per_policy_update", type=int, default=-1,
                        help="If -1, quantity is determined using a formula")
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
    return parser


def create_env(environment, num_states, num_actions, episode_len):
    if environment == 'bandit':
        return BanditEnv(num_actions=num_actions)
    elif environment == 'mdp':
        return MDPEnv(num_states=num_states,
                      num_actions=num_actions,
                      max_episode_length=episode_len)
    else:
        raise NotImplementedError


def create_preprocessing(environment, num_states, num_actions):
    if environment == 'bandit':
        return MABPreprocessing(num_actions=num_actions)
    elif environment == 'mdp':
        return MDPPreprocessing(
            num_states=num_states, num_actions=num_actions)
    else:
        raise NotImplementedError


def create_architecture(
        environment, architecture, input_dim, num_features,
        forget_bias, episode_len, episodes_per_meta_episode
):
    episode_len = 1 if environment == 'bandit' else episode_len
    context_size = episode_len * episodes_per_meta_episode

    if architecture == 'gru':
        return GRU(
            input_dim=input_dim,
            hidden_dim=num_features,
            forget_bias=forget_bias,
            use_ln=True,
            reset_after=True)
    elif architecture == 'snail':
        return SNAIL(
            input_dim=input_dim,
            feature_dim=num_features,
            context_size=context_size,
            use_ln=True)
    elif architecture == 'trxli':
        return TransformerXLI(
            input_dim=input_dim,
            n_layer=12,
            n_head=4,
            d_model=num_features,
            d_head=(num_features // 4))
    elif architecture == 'sparse_transformer':
        return SparseTransformerXLI(
            input_dim=input_dim,
            n_layer=12,
            n_head=4,
            d_model=num_features,
            d_head=(num_features // 4),
            n_context=context_size)
    else:
        raise NotImplementedError


def create_head(num_features, num_actions, head_type):
    if head_type == 'policy':
        return LinearPolicyHead(
            num_features=num_features,
            num_actions=num_actions)
    elif head_type == 'value':
        return LinearValueHead(
            num_features=num_features)
    else:
        raise NotImplementedError


def create_net(
        environment, architecture, num_states, num_actions, num_features,
        forget_bias, episode_len, episodes_per_meta_episode, net_type
):
    preprocessing = create_preprocessing(
        environment=environment,
        num_states=num_states,
        num_actions=num_actions)
    architecture = create_architecture(
        environment=environment,
        architecture=architecture,
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        forget_bias=forget_bias,
        episode_len=episode_len,
        episodes_per_meta_episode=episodes_per_meta_episode)
    head = create_head(
        num_features=architecture.output_dim,
        num_actions=num_actions,
        head_type=net_type)

    if net_type == 'policy':
        return StatefulPolicyNet(
            preprocessing=preprocessing,
            architecture=architecture,
            policy_head=head)
    elif net_type == 'value':
        return StatefulValueNet(
            preprocessing=preprocessing,
            architecture=architecture,
            value_head=head)
    else:
        raise NotImplementedError


def main():
    args = create_argparser().parse_args()
    comm = get_comm()

    # create env.
    env = create_env(
        environment=args.environment,
        num_states=args.num_states,
        num_actions=args.num_actions,
        episode_len=args.episode_len)

    # create learning system.
    policy_net = create_net(
        environment=args.environment,
        architecture=args.architecture,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        forget_bias=args.forget_bias,
        episode_len=args.episode_len,
        episodes_per_meta_episode=args.episodes_per_meta_episode,
        net_type='policy')

    value_net = create_net(
        environment=args.environment,
        architecture=args.architecture,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        forget_bias=args.forget_bias,
        episode_len=args.episode_len,
        episodes_per_meta_episode=args.episodes_per_meta_episode,
        net_type='value')

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
    episode_len = 1 if args.environment == 'bandit' else args.episode_len
    if args.meta_episodes_per_policy_update == -1:
        num_procs = comm.Get_size()
        numer = 240000  # total number of observations per policy improvement
        denom = num_procs * episode_len * args.episodes_per_meta_episode
        meta_episodes_per_policy_update = numer // denom
    else:
        meta_episodes_per_policy_update = args.meta_episodes_per_policy_update

    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        episode_len=episode_len,
        episodes_per_meta_episode=args.episodes_per_meta_episode,
        meta_episodes_per_actor_batch=args.meta_episodes_per_actor_batch,
        meta_episodes_per_policy_update=meta_episodes_per_policy_update,
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
