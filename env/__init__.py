from gymnasium.envs.registration import register
from env.memory import MemoryEnv

__all__ = ["MemoryEnv"]

register(
    id='MiniGrid-MemoryS17Random-v0',
    entry_point='env.memory:MemoryEnv',
    kwargs={'size': 17, 'random_length': True}
)

register(
    id='MiniGrid-MemoryS13Random-v0',
    entry_point='env.memory:MemoryEnv',
    kwargs={'size': 13, 'random_length': True}
)

register(
    id='MiniGrid-MemoryS13-v0',
    entry_point='env.memory:MemoryEnv',
    kwargs={'size': 13}
)

register(
    id='MiniGrid-MemoryS11-v0',
    entry_point='env.memory:MemoryEnv',
    kwargs={'size': 11}
)