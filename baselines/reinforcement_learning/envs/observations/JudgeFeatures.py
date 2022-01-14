import torch
import torch as nn
from collections import defaultdict
import copy

from flatland.envs.step_utils.states import TrainState
from flatland.core.grid.grid4_utils import get_new_position
from ..utils.deadlock_check import check_for_deadlock, get_agent_positions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths


class JudgeFeatures():
    def __init__(self):
        self.state_sz = 7
        self.path_discount = 0.99 # can be decreased

    def reset(self, env):
        self.env = env
        self.cell_to_agents = defaultdict(list)
        self.cell_to_agent_directions = defaultdict(list)
        self.paths = get_shortest_paths(self.env.distance_map)

        self.paths_start = copy.deepcopy(self.paths)
        self.cell_to_agents_before_switch = defaultdict(list)
        for k, v in self.paths_start.items():
            for i, c in enumerate(v):
                if (c.position[0], c.position[1], c.direction) in self.env.obs_builder.tree_obs.greedy_checker.switches:
                    self.paths_start[k] = v[:i + 1]
                    break

        for handle, agent in enumerate(self.env.agents):
            path = self.paths[handle]
            for step in path:
                self.cell_to_agents[step.position].append(handle) 
                self.cell_to_agent_directions[step.position].append(step.direction)
            for step in self.paths_start[handle]:
                self.cell_to_agents_before_switch[step.position].append((handle, step.direction))

        self.VAL = 1. / 100.
        self.NUM_NORM = 1. / 10.

        self.distance = torch.tensor([_get_dist(env, handle) for handle in range(len(self.env.agents))], dtype=torch.float)
        self.load_opposite_direction = torch.zeros(len(self.env.agents))
        self.load_same_direction = torch.zeros(len(self.env.agents))

        self.num_agents_opposite_direction = torch.zeros(len(self.env.agents))
        self.num_agents_same_direction = torch.zeros(len(self.env.agents))
        self.has_deadlocked_on_way = torch.zeros(len(self.env.agents))
        self.num_opposite_agents_before_switch = torch.zeros(len(self.env.agents))

        self.distance *= self.VAL

        self.prev_agent_pos = defaultdict(lambda: (-1, -1))

    def get_many(self, handles):
        return torch.stack([
            self.distance[handles],
            self.load_opposite_direction[handles],
            self.load_same_direction[handles],
            self.num_agents_opposite_direction[handles] * self.NUM_NORM,
            self.num_agents_same_direction[handles] * self.NUM_NORM,
            2 * (self.num_opposite_agents_before_switch[handles] > 0).float() - 1, # has agent before switch
            2 * self.has_deadlocked_on_way[handles] - 1,
        ], dim=1)

    def update_begin(self, handles):
        for handle in handles:
            agent = self.env.agents[handle]
            if agent.state == TrainState.MOVING:
                pos, dir = agent.position, agent.direction
            elif agent.state == TrainState.READY_TO_DEPART:
                pos, dir = agent.initial_position, agent.initial_direction
            else:
                continue

            if self.prev_agent_pos[handle] == pos:
                continue

            for opp_handle, direction in zip(self.cell_to_agents[pos], self.cell_to_agent_directions[pos]):
                if dir == direction:
                    self.num_agents_same_direction[opp_handle] += 1
                else:
                    self.num_agents_opposite_direction[opp_handle] += 1

            for opp_handle, direction in self.cell_to_agents_before_switch[pos]:
                if dir != direction:
                    self.num_opposite_agents_before_switch[opp_handle] += 1

    def update_end(self, handles):
        for handle in handles:
            agent = self.env.agents[handle]
            if agent.state == TrainState.MOVING:
                pos, dir = agent.position, agent.direction
            elif agent.state == TrainState.READY_TO_DEPART:
                pos, dir = agent.initial_position, agent.initial_direction
            else:
                continue

            if self.prev_agent_pos[handle] == pos:
                continue
            self.prev_agent_pos[handle] = pos

            for opp_handle, direction in zip(self.cell_to_agents[pos], self.cell_to_agent_directions[pos]):
                if dir == direction:
                    self.num_agents_same_direction[opp_handle] -= 1
                else:
                    self.num_agents_opposite_direction[opp_handle] -= 1

            for opp_handle, direction in self.cell_to_agents_before_switch[pos]:
                if dir != direction:
                    self.num_opposite_agents_before_switch[opp_handle] -= 1

    def _start_agent(self, handle):
        path = self.paths[handle]
        cur_val = self.VAL
        for step in path:
            (x, y) = step.position
            d = step.direction
            if not (x, y) in self.env.obs_builder.tree_obs.greedy_checker.switches:
                continue
            for opp_handle, direction in zip(self.cell_to_agents[(x, y)], self.cell_to_agent_directions[(x, y)]):
                if direction == d:
                    self.load_same_direction[opp_handle] += cur_val
                else:
                    self.load_opposite_direction[opp_handle] += cur_val
            cur_val *= self.path_discount

    def _finish_agent(self, handle):
        if self.env.agents[handle].state == TrainState.DONE:
            path = self.paths[handle]
            cur_val = self.VAL
            for step in path:
                (x, y) = step.position
                d = step.direction
                if not (x, y) in self.env.obs_builder.tree_obs.greedy_checker.switches:
                    continue
                for opp_handle, direction in zip(self.cell_to_agents[(x, y)], self.cell_to_agent_directions[(x, y)]):
                    if direction == d:
                        self.load_same_direction[opp_handle] -= cur_val
                    else:
                        self.load_opposite_direction[opp_handle] -= cur_val
                cur_val *= self.path_discount
        else:
            path = self.paths[handle]
            cur_val = self.VAL
            for step in path:
                (x, y) = step.position
                d = step.direction
                if not (x, y) in self.env.obs_builder.tree_obs.greedy_checker.switches:
                    continue
                for opp_handle, direction in zip(self.cell_to_agents[(x, y)], self.cell_to_agent_directions[(x, y)]):
                    if direction == d:
                        self.load_same_direction[opp_handle] -= cur_val
                    else:
                        self.load_opposite_direction[opp_handle] -= cur_val
                cur_val *= self.path_discount
            pos = self.env.agents[handle].position
            for opp_handle in self.cell_to_agents[pos]:
                self.has_deadlocked_on_way[opp_handle] = 1


def _get_dist(env, handle):
    agent = env.agents[handle]
    # assert(agent.state == TrainState.READY_TO_DEPART)
    if agent.state.is_off_map_state():
        agent_virtual_position = agent.initial_position
    elif agent.state.is_on_map_state():
        agent_virtual_position = agent.position
    elif agent.state == TrainState.DONE:
        agent_virtual_position = agent.target
    distance_map = env.distance_map.get()
    dist = distance_map[(handle, *agent_virtual_position, agent.direction)]
    return dist
"""
def _build_shortest_path(env, handle):
    agent = env.agents[handle]
    pos = agent.initial_position
    dir = agent.initial_direction

    dist_min_to_target = env.obs_builder.rail_graph.dist_to_target(handle, pos[0], pos[1], dir)

    path = set()
    while dist_min_to_target:
        path.add((*pos, dir))
        possible_transitions = env.rail.get_transitions(*pos, dir)
        for new_dir in range(4):
            if possible_transitions[new_dir]:
                new_pos = get_new_position(pos, new_dir)
                new_min_dist = env.obs_builder.rail_graph.dist_to_target(handle, new_pos[0], new_pos[1], new_dir)
                if new_min_dist + 1 == dist_min_to_target:
                    dist_min_to_target = new_min_dist
                    pos, dir = new_pos, new_dir
                    break

    return path
"""

