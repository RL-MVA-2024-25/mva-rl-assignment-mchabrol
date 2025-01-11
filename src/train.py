from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
#from fast_env import FastHIVPatient
from train_dqn import greedy_action
from train_dqn import DQN_model
import numpy as np
import random
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim


# env = TimeLimit(
#     env=HIVPatient(domain_randomization=True), max_episode_steps=200
# )  # The time wrapper limits the number of steps in an episode at 200.
# # Now is the floor is yours to implement the agent and train it.


class ProjectAgent:

    def act(self, observation):
        s = observation
        a = greedy_action(self.model, s)
        return a
    
    def save(self, path):
        print(f"saving model to {path}")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)

    def load(self):
        path = "model_saved.pt" 
        self.model = DQN_model(hidden_dim = 256)
        # Load model
        model_saved = torch.load("model_saved.pt")
        self.model.load_state_dict(model_saved['model_state_dict'])
        self.model.eval() 
        print(f"Model loaded from {path}")
