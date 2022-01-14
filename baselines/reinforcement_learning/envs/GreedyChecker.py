from collections import defaultdict
import numpy as np

from flatland.envs.agent_utils import TrainState

from flatland.core.grid.grid4_utils import get_new_position
from .utils.deadlock_check import check_for_deadlock, get_agent_positions

class GreedyChecker():
    def __init__(self):
        pass

    def reset(self, env):
        self.env = env
        self.reinit_greedy()

    def on_decision_cell(self, h, w, d):
        return self.decision_cells[h, w, d]

    def on_switch(self, handle):
        agent = self.env.agents[handle]
        return agent.state.is_on_map_state() and (*agent.position, agent.direction) in self.switches

    def greedy_position(self, handle):
        agent = self.env.agents[handle]
        return (not agent.state in (TrainState.DONE, TrainState.READY_TO_DEPART)) \
                and \
                    (not self.on_decision_cell(*agent.position, agent.direction) or agent.malfunction_data["malfunction"] != 0) \
                and \
                    not check_for_deadlock(handle, self.env, get_agent_positions(self.env))


    def reinit_greedy(self):
        self.greedy_way = defaultdict(int)
        rail_env = self.env
        self.location_has_target = set(agent.target for agent in self.env.agents)
        self.switches = set()

        for h in range(rail_env.height):
            for w in range(rail_env.width):
                pos = (h, w)
                transition_bit = bin(self.env.rail.get_full_transitions(*pos))
                total_transitions = transition_bit.count("1")
                if total_transitions > 2:
                    self.switches.add(pos)

        self.target_neighbors = set()
        self.switches_neighbors = set()

        for h in range(rail_env.height):
            for w in range(rail_env.width):
                pos = (h, w)
                for orientation in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, orientation)
                    for ndir in range(4):
                        if possible_transitions[ndir]:
                            nxt = get_new_position(pos, ndir)
                            if nxt in self.location_has_target:
                                self.target_neighbors.add((h, w, orientation))
                            if nxt in self.switches:
                                self.switches_neighbors.add((h, w, orientation))

        self.decision_cells = np.zeros((self.env.height, self.env.width, 4), dtype=np.bool)
        for posdir in self.switches_neighbors.union(self.target_neighbors):
            self.decision_cells[posdir] = 1
        for pos in self.switches.union(self.location_has_target):
            self.decision_cells[pos[0], pos[1], :] = 1

        self.location_has_target_array = np.zeros((self.env.height, self.env.width), dtype=np.bool)
        for pos in self.location_has_target:
            self.location_has_target_array[pos] = 1
        self.location_has_target = self.location_has_target_array
