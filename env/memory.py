from __future__ import annotations

import numpy as np

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Wall
from minigrid.core.constants import COLOR_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

def color_to_ascii(color_name: str) -> str:
    """
    Converts color to it's one letter ascii representation.
    Colors are defined as an ordinal value in minigrid/core/constants.py

    COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
    (ASCII)     (=)        A           B          C            D            E          F
    """
    return chr(COLOR_TO_IDX.get(color_name) + ord("A"))


class MemoryEnv(MiniGridEnv):
    """
    ## Description

    This environment is a memory test. The agent starts in a small room where it
    sees an object. It then has to go through a narrow hallway which ends in a
    split. At each end of the split there is an object, one of which is the same
    as the object in the starting room. The agent has to remember the initial
    object, and go to the matching object at split.

    ## Mission Space

    "go to the matching object at the end of the hallway"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the correct matching object.
    2. The agent reaches the wrong matching object.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of map SxS.

    - `MiniGrid-MemoryS17Random-v0`
    - `MiniGrid-MemoryS13Random-v0`
    - `MiniGrid-MemoryS13-v0`
    - `MiniGrid-MemoryS11-v0`

    """

    def __init__(
        self, size=8, random_length=False, max_steps: int | None = None, **kwargs
    ):
        self.size = size
        self.random_length = random_length

        if max_steps is None:
            max_steps = 5 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "go to the matching object at the end of the hallway"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = np.array((self._rand_int(1, hallway_end + 1), height // 2))
        self.agent_dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        self.grid.set(1, height // 2 - 1, start_room_obj("green"))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0]("green"))
        self.grid.set(*pos1, other_objs[1]("green"))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = "go to the matching object at the end of the hallway"

    def step(self, action):
        if action == Actions.pickup:
            action = Actions.toggle
        obs, reward, terminated, truncated, info = super().step(action)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            terminated = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            terminated = True

        return obs, reward, terminated, truncated, info

    def pprint_grid(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if self.agent_pos is None or self.agent_dir is None or self.grid is None:
            raise ValueError(
                "The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown."
            )

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        # check if self.agent_pos & self.agent_dir is None
        # should not be after env is reset
        if self.agent_pos is None:
            return super().__str__()

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + color_to_ascii(tile.color)
                    else:
                        output += "D" + color_to_ascii(tile.color)
                    continue

                output += OBJECT_TO_STR[tile.type] + color_to_ascii(tile.color)

            if j < self.grid.height - 1:
                output += "\n"

        return output
