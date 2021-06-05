"""
Implements training loop for the bandit agent from Duan et al., 2016
- 'RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning'
"""

from typing import Iterator, Tuple, Dict, Optional, List
from collections import namedtuple, deque
import argparse
from copy import deepcopy

import torch as tc
import numpy as np
from mpi4py import MPI

from rl2.envs.mdp_env import MDPEnv
from rl2.agents.mdp_agent import PolicyNetworkMDP, ValueNetworkMDP
from rl2.utils.comm_util import get_comm, sync_state, sync_grads
from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.constants import ROOT_RANK


TrajSegmentExperience = namedtuple(
    typename='TrajSegmentExperience',
    field_names=[
        'obs', 'acs', 'rews', 'dones',
        'logpacs', 'vpreds', 'tdlam_rets', 'advs',
    ])

TrajSegmentMetrics = Dict[str, List[float]]


@tc.no_grad()
def _traj_segment_generator(
        env: BanditEnv,
        policy_net: PolicyNetworkMAB,
        value_net: ValueNetworkMAB,
        traj_seg_length: int
    ) -> Iterator[Tuple[TrajSegmentExperience, TrajSegmentMetrics]]:
    """
    Returns a stateful generator of trajectory segments.
    Assumes the boundary for each MAB problem aligns with
    traj_seg_length, and resets the env MAB problem after each yield.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network.
        traj_seg_length: timesteps per trajectory segment.

    Returns:
        Iterator of experience, metric tuples.
    """

    traj_seg = TrajSegmentExperience(
        obs=np.array([None for _ in range(traj_seg_length)]),
        acs=np.zeros(traj_seg_length, 'int64'),
        rews=np.zeros(traj_seg_length, 'float32'),
        dones=np.zeros(traj_seg_length, 'float32'),
        logpacs=np.zeros(traj_seg_length, 'float32'),
        vpreds=np.zeros(traj_seg_length+1, 'float32'),
        tdlam_rets=np.zeros(traj_seg_length, 'float32'),
        advs=np.zeros(traj_seg_length, 'float32'),
    )

    current_ep_return = 0.0
    metrics = {
        "ep_returns": []
    }

    t = 0
    h_tm1_policy_net = None
    h_tm1_value_net = None
    a_tm1 = None
    r_tm1 = None
    d_tm1 = None
    o_t = None

    while True:
        if t > 0 and t % traj_seg_length == 0:
            traj_seg.vpreds[-1] = 0.0
            yield traj_seg, metrics
            metrics['ep_returns'] = []
            env.new_mdp()

        if t % traj_seg_length == 0:
            h_tm1_policy_net = policy_net.initial_state(batch_size=1)
            h_tm1_value_net = value_net.initial_state(batch_size=1)
            a_tm1 = np.array([0])
            r_tm1 = np.array([0.0])
            d_tm1 = np.array([1.0])
            o_t = None

        pi_dist_t, h_t_policy_net = policy_net(
            prev_action=tc.LongTensor(a_tm1),
            prev_reward=tc.FloatTensor(r_tm1),
            prev_done=tc.FloatTensor(d_tm1),
            prev_state=h_tm1_policy_net)

        vpred_t, h_t_value_net = value_net(
            prev_action=tc.LongTensor(a_tm1),
            prev_reward=tc.FloatTensor(r_tm1),
            prev_done=tc.FloatTensor(d_tm1),
            prev_state=h_tm1_value_net)

        a_t = pi_dist_t.sample()
        log_prob_a_t = pi_dist_t.log_prob(a_t)

        o_tp1, r_t, done_t, _ = env.step(a_t)

        i = t % traj_seg_length
        traj_seg.obs[i] = o_t
        traj_seg.acs[i] = a_t.squeeze(0).detach().numpy()
        traj_seg.rews[i] = r_t
        traj_seg.dones[i] = done_t
        traj_seg.logpacs[i] = log_prob_a_t.squeeze(0).detach().numpy()
        traj_seg.vpreds[i] = vpred_t.squeeze(0).detach().numpy()

        current_ep_return += r_t
        if i == traj_seg_length-1:
            metrics['ep_returns'].append(current_ep_return)
            current_ep_return = 0.0

        t += 1
        h_tm1_policy_net = h_t_policy_net
        h_tm1_value_net = h_t_value_net
        a_tm1 = np.array([a_t.squeeze(0).detach().numpy()])
        r_tm1 = np.array([r_t])
        d_tm1 = np.array([traj_seg.dones[i]])
        o_t = o_tp1


@tc.no_grad()
def _add_vtarg_and_adv(
        traj_seg: TrajSegmentExperience,
        gamma: float,
        lam: float
    ) -> TrajSegmentExperience:
    """
    Compute td lambda returns and generalized advantage estimates.

    Args:
        traj_seg: trajectory segment.
        gamma: discount factor.
        lam: GAE decay parameter.

    Returns:
        traj_seg
    """
    T = len(traj_seg.acs)
    for t in reversed(range(1, T+1)):  # T, ..., 1.
        r_t = traj_seg.rews[t-1]
        V_t = traj_seg.vpreds[t-1]
        V_tp1 = traj_seg.vpreds[t]
        A_tp1 = traj_seg.advs[t] if t < T else 0.0
        mask_t = 1.0 if t < T else 0.0
        delta_t = -V_t + r_t + gamma * mask_t * V_tp1
        A_t = delta_t + gamma * lam * mask_t * A_tp1
        traj_seg.advs[t-1] = A_t

    traj_seg.tdlam_rets[:] = traj_seg.vpreds[0:-1] + traj_seg.advs
    return traj_seg


def _compute_losses(
        policy_net: PolicyNetworkMDP,
        value_net: ValueNetworkMDP,
        traj_segs: List[TrajSegmentExperience],
        clip_param: float,
        ent_coef: float
    ) -> Dict[str, tc.Tensor]:
    """
    Computes the losses for PPO using the policy network and value network.

    Args:
        policy_net: policy network.
        value_net: value network.
        traj_segs: list of trajectory segments w/ updated advs & td lam returns.
        clip_param: clip parameter.
        ent_coef: entropy coefficient.

    Returns:
        loss_dict: a dictionary of losses.
    """
    def get_tensor(field, dtype=None):
        mb_field = np.stack(
            list(map(lambda seg: getattr(seg, field), traj_segs)),
            axis=0)
        if dtype == 'long':
            return tc.LongTensor(mb_field)
        return tc.FloatTensor(mb_field)

    mb_acs = get_tensor('acs', 'long')
    mb_rews = get_tensor('rews')
    mb_dones = get_tensor('dones')
    mb_logpacs = get_tensor('logpacs')
    mb_advs = get_tensor('advs')
    mb_tdlam_rets = get_tensor('tdlam_rets')

    # we are going to manually loop over the timesteps here,
    # for backprop thru time.
    B = len(traj_segs)
    T = len(traj_segs[0].acs)
    t = 1
    h_tm1_policy_net = policy_net.initial_state(batch_size=B)
    h_tm1_value_net = value_net.initial_state(batch_size=B)
    a_tm1 = tc.zeros(dtype=tc.int64, size=(B,))
    r_tm1 = tc.zeros(dtype=tc.float32, size=(B,))
    d_tm1 = tc.ones(dtype=tc.float32, size=(B,))

    entropies = []
    vpreds_new = []
    logpacs_new = []
    while t < T+1:
        pi_dist_t, h_t_policy_net = policy_net(
            prev_action=a_tm1,
            prev_reward=r_tm1,
            prev_done=d_tm1,
            prev_state=h_tm1_policy_net)

        vpred_t, h_t_value_net = value_net(
            prev_action=a_tm1,
            prev_reward=r_tm1,
            prev_done=d_tm1,
            prev_state=h_tm1_value_net)

        ent_t = pi_dist_t.entropy()
        entropies.append(ent_t)

        a_t = mb_acs[:, t-1]
        r_t = mb_rews[:, t-1]
        d_t = mb_dones[:, t-1]

        logprob_a_t_new = pi_dist_t.log_prob(a_t)
        logpacs_new.append(logprob_a_t_new)
        vpreds_new.append(vpred_t)

        t += 1
        h_tm1_policy_net = h_t_policy_net
        h_tm1_value_net = h_t_value_net
        a_tm1 = a_t
        r_tm1 = r_t
        d_tm1 = d_t

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


def _train(
        env: MDPEnv,
        policy_net: PolicyNetworkMDP,
        value_net: ValueNetworkMDP,
        policy_optimizer: tc.optim.Optimizer,
        value_optimizer: tc.optim.Optimizer,
        policy_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        value_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        args: argparse.Namespace,
        env_steps_so_far: int
    ) -> None:
    """
    Train the agent by PPO to solve multi-armed bandit problems
    sampled from the distribution used in Duan et al., 2016.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network,
        policy_optimizer: policy optimizer.
        value_optimizer: value optimizer.
        policy_scheduler: policy lr scheduler.
        value_scheduler: value lr scheduler.
        args: argparsed args.
        env_steps_so_far: environment steps so far.

    Returns:
        None
    """
    comm = get_comm()
    traj_seg_gen = _traj_segment_generator(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        traj_seg_length=args.traj_seg_length)

    policy_improvement_step = 0
    collection_step = 0
    traj_segs = []
    ep_returns = deque(maxlen=1000)
    while policy_improvement_step < args.max_policy_iterations:
        # update traj_segs with new seg.
        traj_seg, local_metrics = next(traj_seg_gen)
        traj_seg, local_metrics = deepcopy(traj_seg), deepcopy(local_metrics)
        traj_seg = _add_vtarg_and_adv(traj_seg, gamma=args.gamma, lam=args.lam)
        traj_segs.append(traj_seg)

        # update episode returns after each trajectory.
        global_ep_returns = comm.allgather(local_metrics['ep_returns'])
        global_ep_returns = [x for loc in global_ep_returns for x in loc]
        ep_returns.extend(global_ep_returns)

        collection_step += 1

        if collection_step % args.optim_nseg_per_actor == 0:
            for opt_epoch in range(args.optim_epochs):
                idxs = np.random.permutation(args.optim_nseg_per_actor)
                for i in range(0, args.optim_nseg_per_actor, args.optim_nseg_per_gradstep):
                    mb_idxs = idxs[i:i+args.optim_nseg_per_gradstep]
                    mb_traj_segs = [traj_segs[idx] for idx in mb_idxs]
                    losses = _compute_losses(
                        policy_net=policy_net,
                        value_net=value_net,
                        traj_segs=mb_traj_segs,
                        clip_param=args.clip_param,
                        ent_coef=args.ent_coef)

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

                global_losses = {}
                for name in losses:
                    loss_sum = comm.allreduce(losses[name], op=MPI.SUM)
                    loss_avg = loss_sum / comm.Get_size()
                    global_losses[name] = loss_avg

                if comm.Get_rank() == ROOT_RANK:
                    print(f"pol iter {policy_improvement_step}, opt_epoch: {opt_epoch}...")
                    for name, value in global_losses.items():
                        print(f"\t{name}: {value:>0.6f}")

            if comm.Get_rank() == ROOT_RANK:
                print("-" * 100)
                print(f"mean_ep_return: {np.mean(ep_returns)}")
                print("-" * 100)

            # we completed one policy improvement step.
            policy_improvement_step += 1
            traj_segs = []

            # maybe checkpoint.
            if True:
                if comm.Get_rank() == 0:
                    save_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        model_name=f"{args.model_name}/policy_net",
                        model=policy_net,
                        optimizer=policy_optimizer,
                        scheduler=policy_scheduler,
                        steps=env_steps_so_far)

                    save_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        model_name=f"{args.model_name}/value_net",
                        model=value_net,
                        optimizer=value_optimizer,
                        scheduler=value_scheduler,
                        steps=env_steps_so_far)

        # after each collection step, update env steps.
        # then collect more experience.
        env_steps_so_far += args.traj_seg_length * comm.Get_size()


def _get_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script.""")
    parser.add_argument("--max_policy_iterations", type=int, default=1000)
    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--traj_seg_length", type=int, default=10)
    parser.add_argument("--optim_epochs", type=int, default=4)
    parser.add_argument("--optim_nseg_per_actor", type=int, default=30000//10)
    parser.add_argument("--optim_nseg_per_gradstep", type=int, default=60)
    parser.add_argument("--optim_stepsize", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--clip_param", type=float, default=0.10)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--monitoring_dir", type=str, default='monitoring')
    parser.add_argument("--asset_dir", type=str, default='assets')
    parser.add_argument("--experiment_seed", type=int, default=0)
    return parser


def main():   # pylint: disable=C0116
    args = _get_argparser().parse_args()
    env = MDPEnv(
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_ep_length=10)

    policy_net = PolicyNetworkMDP(
        num_states=args.num_states,
        num_actions=args.num_actions)
    value_net = ValueNetworkMDP(
        num_states=args.num_states,
        num_actions=args.num_actions)

    policy_optimizer = tc.optim.Adam(
        params=policy_net.parameters(),
        lr=args.optim_stepsize,
        eps=1e-5)
    value_optimizer = tc.optim.Adam(
        params=value_net.parameters(),
        lr=args.optim_stepsize,
        eps=1e-5)

    policy_scheduler = None
    value_scheduler = None

    comm = get_comm()
    env_steps_so_far = 0
    if comm.Get_rank() == 0:
        steps_policy = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/policy_net",
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None)

        steps_value = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/value_net",
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None)

        if steps_policy != steps_value:
            raise RuntimeError(
                "Policy steps not equal to value steps in latest checkpoint!")
        env_steps_so_far = steps_policy

    env_steps_so_far = comm.bcast(env_steps_so_far, root=ROOT_RANK)

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

    _train(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        args=args,
        env_steps_so_far=env_steps_so_far
    )


if __name__ == '__main__':
    main()
