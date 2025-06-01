from __future__ import annotations

import gymnasium as gym
from minigrid.core.actions import Actions
from memory import MemoryEnv


class ManualTextControl:
    def __init__(
        self,
        env: MemoryEnv,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the text-based game loop"""
        self.reset(self.seed)
        self.env.render()  # Initial render

        while not self.closed:
            try:
                command = input("Enter action (left, right, forward, toggle, pickup, drop, done, reset, exit): ").strip().lower()
                if command == "exit":
                    self.env.close()
                    self.closed = True
                    print("Exiting.")
                    break
                elif command == "reset":
                    self.reset(self.seed)
                    self.env.render()
                elif command in self.key_to_action():
                    self.step(self.key_to_action()[command])
                else:
                    print(f"Unknown command: {command}")
            except EOFError: # Handle Ctrl+D or unexpected EOF
                print("\nExiting due to EOF.")
                self.env.close()
                self.closed = True
                break
            except KeyboardInterrupt: # Handle Ctrl+C
                print("\nExiting due to user interrupt.")
                self.env.close()
                self.closed = True
                break


    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"Step={self.env.unwrapped.step_count}, Reward={reward:.2f}")

        if terminated:
            print("Terminated!")
            self.env.render() # Render final state
            print("Resetting environment...")
            self.reset(self.seed)
        elif truncated:
            print("Truncated!")
            self.env.render() # Render final state
            print("Resetting environment...")
            self.reset(self.seed)
        else:
            self.env.render() # Render after each step

    def reset(self, seed=None):
        print(f"Resetting environment with seed: {seed}")
        self.env.reset(seed=seed)
        # No initial render here, start() will call it or step() will.

    def key_to_action(self):
        return {
            "left": Actions.left,
            "right": Actions.right,
            "forward": Actions.forward,
            "toggle": Actions.toggle,
            "pickup": Actions.pickup,
            "drop": Actions.drop,
            "done": Actions.done,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="Gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MemoryS13Random-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )

    args = parser.parse_args()

    # Determine size and random_length from args.env_id
    size_str_parts = args.env_id.split('S')
    if len(size_str_parts) > 1:
        size_str = size_str_parts[1].split('Random')[0].split('-')[0]
        try:
            size = int(size_str)
        except ValueError:
            print(f"Could not parse size from env_id: {args.env_id}. Using default size 13.")
            size = 13  # Default size
    else:
        print(f"Could not parse size from env_id: {args.env_id}. Using default size 13.")
        size = 13 # Default size

    random_length = "Random" in args.env_id

    # Create the MemoryEnv directly with render_mode="text"
    print(f"Creating MemoryEnv: size={size}, random_length={random_length}, render_mode='text'")
    env = MemoryEnv(
        size=size,
        random_length=random_length,
        render_mode="text",  # Key change for text-based rendering
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
    )
    # No wrappers like RGBImgPartialObsWrapper or ImgObsWrapper needed for text mode.

    manual_control = ManualTextControl(env, seed=args.seed)
    print("Starting text-based manual control. Type 'exit' to quit.")
    manual_control.start()
