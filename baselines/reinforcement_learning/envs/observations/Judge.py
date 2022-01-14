from flatland.envs.step_utils.states import TrainState
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from .JudgeNetwork import JudgeNetwork
from .JudgeFeatures import JudgeFeatures
from ..utils.deadlock_check import check_for_deadlock, get_agent_positions

class Judge():
    def __init__(self, lr, batch_size, optimization_epochs, device):
        self.obs_builder = JudgeFeatures()
        self.device = device

        self.net = JudgeNetwork(self.obs_builder.state_sz).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.epochs = optimization_epochs

    def reset(self, env):
        self.env = env
        self.obs_builder.reset(env)
        self.ready_to_depart = [0] * len(self.env.agents)
        # self.send_more = self.window_size

        self.sent_priorities = torch.empty(len(self.env.agents))
        self.sent_states = torch.empty((len(self.env.agents), self.obs_builder.state_sz))

        self.cur_threshold = 0.95
        self.timer = 0
        self.prev_check = -1e9
        self.end_time = torch.empty(len(self.env.agents))
        self.all_out = False
        self.active_agents = set()
        self.valid_states = [TrainState.MOVING, TrainState.READY_TO_DEPART, TrainState.STOPPED, TrainState.MALFUNCTION]
        
        self.priority_timer = 0
        self.last_priorities_update = -1e9
        self.priorities = torch.zeros(len(self.env.agents), dtype=torch.float)
        self.observations = torch.zeros((len(self.env.agents), self.obs_builder.state_sz), dtype=torch.float)

    def get_rollout(self):
        used_handles = [handle for handle in self.active_agents] # [handle for handle in range(len(self.env.agents)) if self.ready_to_depart[handle] != 0]
        target = torch.tensor([1. if self.env.agents[handle].state == TrainState.DONE else 0. for handle in used_handles])
        states = self.sent_states[used_handles]
        return target, states

    # TODO add a replay buffer or something for the sake of data efficiency
    def optimize(self, rollout):
        target, states = rollout
        n = len(target)

        target, states = target.to(self.device).repeat(self.epochs+1), states.to(self.device).repeat((self.epochs+1,1))
        permutation = torch.randperm(len(target))
        target, states = target[permutation], states[permutation]

        sum_loss = 0
        for l in range(0, n*self.epochs, self.batch_size):
            r = min(len(target), l + self.batch_size)
            probs = self.net(states[l:r]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(probs, target[l:r])
            sum_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss = sum_loss
        print("Judge Loss: {}".format(sum_loss))
        return {"loss": sum_loss}


    def get_batch(self, handles):
        with torch.no_grad():
            observations = self.obs_builder.get_many(handles)
            priorities = self.net(observations).squeeze(1)
        return observations, priorities.detach().cpu()

    def _get_observations(self, handles):
        return self.obs_builder.get_many(handles)

    def _get_priorities(self, observations):
        with torch.no_grad():
            priorities = self.net(observations).squeeze(1)
        return priorities.detach().cpu()

    def calc_priorities(self, handles):
        self.observations[handles] = self._get_observations(handles)
        self.priorities[handles] = self._get_priorities(self.observations[handles])
        return self.priorities[handles], self.observations[handles]

    def update(self):
        # update agents that are done
        self.update_finished()
        # update agents that have just started
        self.update_started()
        
        # TODO might be good to also differentiate agents that have malfunction
        
        # TODO might be slow on big envs too
        self.obs_builder.update_begin(self.active_agents)
        
        valid_handles = [agent.handle for agent in self.env.agents if agent.state in self.valid_states]
        _, observations = self.calc_priorities(valid_handles)
        # self.priorities[valid_handles] = priorities
        # print(self.priorities)
        for agent in self.env.agents:
            if agent.handle not in valid_handles:
                self.priorities[agent.handle] = 0
        
        for handle, obs in zip(valid_handles, observations):
            self.sent_states[handle] = obs
            
        self.obs_builder.update_end(self.active_agents)

    def update_finished(self):
        for handle in list(self.active_agents):
            if (self.env.agents[handle].state == TrainState.DONE \
                    or check_for_deadlock(handle, self.env, get_agent_positions(self.env))):
                self._finish_agent(handle)
                
    def _finish_agent(self, handle):
        self.active_agents.remove(handle)
        self.obs_builder._finish_agent(handle)
    
    def update_started(self):
        for agent in self.env.agents:
            if agent.handle not in list(self.active_agents):
                if (agent.state in self.valid_states):
                    self._start_agent(agent.handle)
    
    def _start_agent(self, handle):
        self.active_agents.add(handle)
        self.obs_builder._start_agent(handle)

    def update_net_params(self, net_params):
        self.net.load_state_dict(net_params)

    def get_net_params(self, device=None):
        state_dict = self.net.state_dict()
        if device is not None and device != self.device:
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        return state_dict

    def load_judge(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)

    def save_judge(self, filename):
        state_dict = self.get_net_params(device=torch.device("cpu"))
        torch.save(state_dict, filename + ".judge")




