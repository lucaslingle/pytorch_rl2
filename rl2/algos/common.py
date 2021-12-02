"""
Implements common algorithmic components for training
stateful meta-reinforcement learning agents.
"""

import torch as tc
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet


class MetaEpisode:
    def __init__(self, num_timesteps, dummy_obs):
        self.horizon = num_timesteps
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
        meta_episode_len: int
    ) -> MetaEpisode:
    """
    Generates a meta-episode: a sequence of episodes concatenated together,
    with decisions being made by a recurrent agent with state preserved
    across episode boundaries.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network.
        meta_episode_len: timesteps per meta-episode.

    Returns:
        meta_episode: an instance of the meta-episode class.
    """

    env.new_env()
    meta_episode = MetaEpisode(
        num_timesteps=meta_episode_len,
        dummy_obs=env.reset())

    o_t = np.array([env.reset()])
    a_tm1 = np.array([0])
    r_tm1 = np.array([0.0])
    d_tm1 = np.array([1.0])
    h_tm1_policy_net = policy_net.initial_state(batch_size=1)
    h_tm1_value_net = value_net.initial_state(batch_size=1)

    for t in range(0, meta_episode_len):
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

        o_tp1, r_t, done_t, _ = env.step(
            action=a_t.squeeze(0).detach().numpy(),
            auto_reset=True)

        meta_episode.obs[t] = o_t[0]
        meta_episode.acs[t] = a_t.squeeze(0).detach().numpy()
        meta_episode.rews[t] = r_t
        meta_episode.dones[t] = float(done_t)
        meta_episode.logpacs[t] = log_prob_a_t.squeeze(0).detach().numpy()
        meta_episode.vpreds[t] = vpred_t.squeeze(0).detach().numpy()

        o_t = np.array([o_tp1])
        a_tm1 = np.array([meta_episode.acs[t]])
        r_tm1 = np.array([meta_episode.rews[t]])
        d_tm1 = np.array([meta_episode.dones[t]])
        h_tm1_policy_net = h_t_policy_net
        h_tm1_value_net = h_t_value_net

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


def huber_func(y_pred, y_true, delta=1.0):
    a = y_pred-y_true
    a_abs = tc.abs(a)
    a2 = tc.square(a)
    terms = tc.where(
        tc.less(a_abs, delta * tc.ones_like(a2)),
        0.5 * a2,
        delta * (a_abs - 0.5 * delta)
    )
    return terms
