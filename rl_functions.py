import numpy as np
import torch
import torch.nn as nn
import water_shooting
import json
import math
import random


def mlp(sizes, activation, output_activation=nn.Identity):
    """Generates a basic MLP"""
    layers = []
    for j in range(len(sizes)-2):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    
    return nn.Sequential(*layers)


class MLPCritic(nn.Module):
    def __init__():
        #TODO: create Critic model
        pass

class MLPActor(nn.Module):
    def __init__():
        #TODO: create Actor model
        pass

class MLPActorCritic(nn.Module):
    """Combine policy and value function nns"""

    def __init__(self, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        self.pi = MLPActor() #TODO fill with params
        self.v = MLPCritic() #TODO fill with params
    
    def step(self, state):
        with torch.no_grad():
            pi = self.pi_distribution(torch.as_tensor(state, dtype=torch.float))
            act = pi.sample() # pick a policy based on the probability distribution of π
            #TODO update Actor
            act.step(state)
        return act

class DummyActorCritic(nn.Module):
    """Combine policy and value function nns"""

    def __init__(*args):
        return
    
    def step(self, state):
        return random.choice(['a','d','LMB'])

class Agent:

    def __init__(self, env):
        self.env = env
        self.width = 64 #width of network
        self.l = 2 #number of layers in the network
        self.ac = DummyActorCritic()

    #We'll need to update pi, v, we need a train function
    def play(self):
        self.env.play(self.ac)

def main():
    with open('env_setting.json', 'r') as file:
        setting = json.load(file)

    setting.update({'jet_angular_speed': (setting['jet_speed'] * math.pi)})
    env = water_shooting
    agent = Agent(env)
    agent.play()


if __name__ == "__main__":
    main()