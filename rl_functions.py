import numpy as np
import torch
import torch.nn as nn
import water_shooting
import json
import math
import random
import scipy.signal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import tqdm

class EncDecoder(nn.Module):
    def __init__(self, big_state, small_state):
        super().__init__()


def mlp(sizes, activation, output_activation=nn.Identity):
    """Generates a basic MLP"""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    
    return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_pro_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_pro_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    """Combine policy and value function nns"""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation) #TODO fill with params
        self.v = MLPCritic(obs_dim, hidden_sizes, activation) #TODO fill with params
    
    def step(self, state):
        with torch.no_grad():
            pi = self.pi._distribution(torch.as_tensor(state, dtype=torch.float))
            act = pi.sample() # pick a policy based on the probability distribution of π
            with torch.no_grad():
                pi = self.pi._distribution(torch.as_tensor(state, dtype=torch.float32))
                act = pi.sample()
                logp = self.pi._log_pro_from_distribution(pi,act)
                val = self.v(torch.as_tensor(state, dtype=torch.float32))

        return act.item(), val.item(), logp.item()

class VPGBuffer:
    """Buffer to store trajectories."""

    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.phi_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Append a single timestep to the buffer."""

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_traj(self, last_val=0):
        """Call after a trajectory ends."""

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.phi_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews[:-1], self.gamma)

        self.path_start_idx = self.ptr

    def get(self):
        """Call after an epoch ends. Resets pointers and returns the buffer contents."""

        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        self.phi_buf = (self.phi_buf - np.mean(self.phi_buf)) / np.std(self.phi_buf)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    phi=self.phi_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class Agent:

    def __init__(self, env):
        self.env = env
        self.hid = 64  # Layer width of networks
        self.l = 2  # Layer number of networks
        self.obs_dim = 4002
        # Initialize Actor-Critic
        self.ac = MLPActorCritic(self.obs_dim, hidden_sizes=[self.hid] * self.l, act_dim=16)

        pi_lr = 3e-3
        vf_lr = 1e-3

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    def pi_update(self, data):
        """Use the data from the buffer to update the policy."""

        obs = data['obs']
        act = data['act']
        phi = data['phi']

        self.pi_optimizer.zero_grad()

        _, logp = self.ac.pi(obs, act)
        loss = (phi * logp).mean()
        loss.backward()
        self.pi_optimizer.step()

    def v_update(self, data):
        """Use the data from the buffer to update the value function."""

        obs = data['obs']
        ret = data['ret']

        self.v_optimizer.zero_grad()

        for update in range(100):
            val = self.ac.v(obs)
            loss = nn.functional.mse_loss(val, ret)
            loss.backward()
            self.v_optimizer.step()

    def train(self):
        """Main training loop."""

        obs_dim = [4002]
        act_dim = []
        steps_per_epoch = 2000
        epochs = 24
        max_ep_len = 2000
        gamma = 0.99
        lam = 0.97
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the environment
        self.env.reset()
        state, ep_ret, ep_len = self.env.get_state(), 0, 0
        state = state.flatten()
        # Main loop: collect experience in env and update each epoch
        for epoch in range(epochs):
            ep_returns = []

            for t in tqdm.tqdm(range(steps_per_epoch), "Computing the steps for the current epoch"):
                state = state.flatten()
                a, v, logp = self.ac.step(torch.from_numpy(state))
                next_state, r, terminal = self.env.transition(a)
                ep_ret += r
                ep_len += 1

                # Log transition
                buf.store(state, a, r, v, logp)

                # Update state
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # If trajectory didn't reach terminal state, bootstrap value target
                    state = state.flatten()
                    if epoch_ended:
                        _, v, _ = self.ac.step(torch.from_numpy(state))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret) # Only store return when episode ended
                    buf.end_traj(v)

                    self.env.reset()
                    state, ep_ret, ep_len = self.env.get_state(), 0, 0


            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            if len(ep_returns) == 0:
                print(f"Epoch: {epoch+1}/{epochs}, all episodes exceeded max_ep_len")
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

            # Update the policy and value function
            data = buf.get()
            self.pi_update(data)
            self.v_update(data)
            torch.save(self.ac, "params.pt")
        return True

    def get_action(self, obs):
        """Sample an action from the policy."""

        return self.ac.step(obs)[0]


class DummyActorCritic(nn.Module):
    """He do be stoopid but he
    everyone can play the game
    """

    def __init__(*args):
        return
    
    def step(self, state):
        return random.choice(['a','d','LMB'])

def main():
    with open('env_setting.json', 'r') as file:
        setting = json.load(file)
    setting.update({'jet_angular_speed': (setting['jet_speed'] * math.pi)})
    env = water_shooting
    agent = Agent(env)
    should_train = input("Do I train the model? (y/n)")
    if should_train == 'y':
        agent.train()
    else:
        agent.ac = torch.load("params.pt")
    episode_length = 3
    n_eval = 5
    returns = []
    print("Evaluating the agent...")

    for i in range(n_eval):
        env.reset()
        state, ep_ret, ep_len = env.get_state(), 0, 0
        state = state.flatten()
        cumulative_return = 0
        terminal = False
        env.reset()
        for t in range(episode_length):
            state = state.flatten()
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
    torch.save(agent.ac, "params.pt")

    env.play(ac=agent)



if __name__ == "__main__":
    main()