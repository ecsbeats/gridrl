import dspy
import mlflow
from typing import List, Tuple, Optional, Any
from pydantic import BaseModel, Field

def setup_mlflow():
    mlflow.dspy.autolog()
    mlflow.set_experiment("dspy-test")

def configure_dspy() -> dspy.LM:
    lm = get_lm()
    dspy.configure(lm=lm)
    return lm

def get_lm() -> dspy.LM:
    lm = dspy.LM(model="openai/gemma3:27b",
                api_key="X",
                max_tokens=128000,
                api_base="http://localhost:11434/v1",
                model_type="chat")
    return lm

# Pydantic Models for Game State and Memory
class TileDescription(BaseModel):
    raw_chars: str = Field(..., description="Two-character string representing the tile as in pprint_grid.")
    # Potentially add more parsed fields later, e.g., object_type, color, state

class GameState(BaseModel):
    grid_representation: str = Field(..., description="ASCII string representation of the full game grid.")
    agent_pos: Tuple[int, int] = Field(..., description="Agent's current (x, y) position.")
    agent_dir: int = Field(..., description="Agent's current direction (0: right, 1: down, 2: left, 3: up).")
    mission: str = Field(..., description="The mission for the agent.")
    carrying: Optional[str] = Field(None, description="Description of the object the agent is carrying, if any (e.g., 'green key').")
    step_count: int = Field(..., description="Current step count in the episode.")
    max_steps: int = Field(..., description="Maximum steps allowed in the episode.")
    # MemoryEnv specific fields like success_pos and failure_pos are removed
    # as they represent privileged information not directly observable by the agent.
    # The agent should infer goal locations from the mission and grid.

class MemoryEntry(BaseModel):
    step: int
    observation: GameState
    action_taken: Optional[str] = None # Action taken after this observation
    reward_received: Optional[float] = None
    llm_thought_process: Optional[str] = None
    llm_action_prediction: Optional[str] = None # Action predicted by LLM for this state

# Memory Buffer
memory_buffer: List[MemoryEntry] = []

# Action mapping (optional, depends on how game env expects actions)
ACTION_MAP = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}
# Reverse map for converting agent's string output to something the env might use
# or for logging.
ACTION_STR_TO_ENUM = {v: k for k, v in ACTION_MAP.items()} # Example if env uses numeric actions

# Mock Game Environment
class MockMiniGridEnv:
    def __init__(self, grid_size=(6, 6), max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # 0: right, 1: down, 2: left, 3: up
        self.carrying = None
        # Example grid: W=Wall, F=Floor, K=Key, G=Goal. Color chars: R,G,B,Y,P,F(grey)
        self.grid = [
            ["WF", "WF", "WF", "WF", "WF", "WF"],
            ["WF", "  ", "  ", "  ", "KB", "WF"], # KB = green key
            ["WF", "  ", "  ", "  ", "  ", "WF"],
            ["WF", "  ", "  ", "GA", "  ", "WF"], # GA = red goal
            ["WF", "  ", "  ", "  ", "  ", "WF"],
            ["WF", "WF", "WF", "WF", "WF", "WF"],
        ]
        self.mission = "Pick up the green key (KB) and go to the red goal (GA)."
        self.last_action_status = ""

    def _get_agent_char(self):
        return {0: ">>", 1: "VV", 2: "<<", 3: "^^"}[self.agent_dir]

    def _update_grid_representation(self) -> str:
        grid_str_list = []
        for r_idx, row in enumerate(self.grid):
            row_str = []
            for c_idx, cell in enumerate(row):
                if (r_idx, c_idx) == self.agent_pos:
                    row_str.append(self._get_agent_char())
                else:
                    row_str.append(cell)
            grid_str_list.append("".join(row_str))
        return "\\n".join(grid_str_list)

    def get_pydantic_game_state(self) -> GameState:
        return GameState(
            grid_representation=self._update_grid_representation(),
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            mission=self.mission,
            carrying=self.carrying,
            step_count=self.current_step,
            max_steps=self.max_steps,
        )

    def step(self, action_str: str) -> Tuple[GameState, float, bool, bool, dict]:
        self.current_step += 1
        reward = -0.1  # Small penalty for each step
        terminated = False
        truncated = False
        info = {"action_status": "success"}
        self.last_action_status = ""

        action = action_str.lower().strip()
        r, c = self.agent_pos

        if action == "forward":
            nr, nc = r, c
            if self.agent_dir == 0: nc += 1    # Right
            elif self.agent_dir == 1: nr += 1  # Down
            elif self.agent_dir == 2: nc -= 1  # Left
            elif self.agent_dir == 3: nr -= 1  # Up

            if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1] and \
               not self.grid[nr][nc].startswith("W"): # Not a wall
                self.agent_pos = (nr, nc)
                self.last_action_status = "Moved forward."
            else:
                self.last_action_status = "Bumped into a wall or boundary."
                info["action_status"] = "bumped"
                reward -= 0.5 # Penalty for bumping
        elif action == "left":
            self.agent_dir = (self.agent_dir - 1 + 4) % 4
            self.last_action_status = "Turned left."
        elif action == "right":
            self.agent_dir = (self.agent_dir + 1) % 4
            self.last_action_status = "Turned right."
        elif action == "pickup":
            current_cell_content = self.grid[r][c]
            if current_cell_content == "KB" and not self.carrying: # Green Key
                self.carrying = "green key"
                self.grid[r][c] = "  " # Remove key from grid
                self.last_action_status = "Picked up green key."
                reward += 1.0
            else:
                self.last_action_status = f"Nothing to pickup or already carrying. Cell: {current_cell_content}"
                info["action_status"] = "pickup_failed"
        elif action == "drop":
            if self.carrying:
                # For simplicity, just drop it on current square if empty
                if self.grid[r][c] == "  ":
                    if self.carrying == "green key": self.grid[r][c] = "KB"
                    self.last_action_status = f"Dropped {self.carrying}."
                    self.carrying = None
                else:
                    self.last_action_status = "Cannot drop here, cell occupied."
                    info["action_status"] = "drop_failed_occupied"
            else:
                self.last_action_status = "Nothing to drop."
                info["action_status"] = "drop_failed_empty_handed"
        elif action == "toggle":
            # Example: toggle a door, not implemented in this simple grid
            self.last_action_status = "Toggle action attempted (not fully implemented)."
        elif action == "done":
            if self.grid[r][c] == "GA" and self.carrying == "green key": # Red Goal
                self.last_action_status = "Task completed successfully at Goal!"
                reward += 10.0
                terminated = True
            else:
                self.last_action_status = "Task marked done, but conditions not met."
                reward -= 2.0 # Penalty for incorrect 'done'
                terminated = True # End episode on 'done' regardless
        else:
            self.last_action_status = f"Unknown action: {action}"
            info["action_status"] = "unknown_action"

        if self.current_step >= self.max_steps:
            truncated = True
            self.last_action_status += " Max steps reached."

        info["last_action_status"] = self.last_action_status
        new_game_state = self.get_pydantic_game_state()
        return new_game_state, reward, terminated, truncated, info

# DSPy Agent Definition
class GamePlayingAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define a signature for the agent
        # It takes the current game state and a summary of the memory buffer (scratchpad)
        # It outputs the next action to take and its thought process.
        self.predict_action = dspy.ChainOfThought(PlayerSignature)

    def forward(self, current_game_state: GameState, memory: str) -> dspy.Prediction:
        # The memory could be the last N entries, or a condensed summary.
        # For now, we'll pass it as a string.
        prediction = self.predict_action(
            current_grid=current_game_state.grid_representation,
            agent_pos=str(current_game_state.agent_pos),
            agent_dir=str(current_game_state.agent_dir),
            mission=current_game_state.mission,
            carrying=str(current_game_state.carrying),
            steps_taken=str(current_game_state.step_count),
            memory=memory
        )
        return prediction

class PlayerSignature(dspy.Signature):
    """Predict the next best action to take in a grid-based game to achieve a mission, considering past actions and observations.
The grid is represented by a string where each cell is two characters.
Agent representation: >> (right), VV (down), << (left), ^^ (up).
Object representation: First char is type (W:Wall, F:Floor, D:Door, K:Key, A:Ball, B:Box, G:Goal, V:Lava), second char is color (A:red, B:green, C:blue, D:purple, E:yellow, F:grey).
Examples: WF (grey Wall), KB (green Key), AB (green Ball). Empty cells are '  '. Doors: __ (open), DA (closed red door), LA (locked red door).
"""
    current_grid: str = dspy.InputField(desc="The current ASCII representation of the game grid. Each cell is two characters. Agent: >>, VV, <<, ^^. Walls (e.g., grey): WF. Keys (e.g., green): KB. Balls (e.g., green): AB. Empty: '  '.")
    agent_pos: str = dspy.InputField(desc="Agent's current (x,y) position, e.g., (1,1).")
    agent_dir: str = dspy.InputField(desc="Agent's current direction (0: right, 1: down, 2: left, 3: up).")
    mission: str = dspy.InputField(desc="The objective the agent needs to achieve, e.g., 'go to the matching object at the end of the hallway'.")
    carrying: str = dspy.InputField(desc="What the agent is currently carrying (e.g., 'green key' or 'None').")
    steps_taken: str = dspy.InputField(desc="Number of steps taken so far in this episode.")
    memory: str = dspy.InputField(desc="A summary of recent observations, actions, and outcomes from the scratchpad/memory.")

    next_action: str = dspy.OutputField(desc="The single best action to take next. Choose from: left, right, forward, toggle, pickup, drop, done.")
    thought_process: str = dspy.OutputField(desc="Briefly explain your reasoning for choosing this action based on the mission, current state, and memory.")


# Mocked Implementation
def run_dspy_agent_turn(
    game_env_facade: MockMiniGridEnv, # Updated type hint
    dspy_agent: GamePlayingAgent,
    lm: dspy.LM
):
    # 1. Get current game state from game_env_facade
    current_game_state_from_env = game_env_facade.get_pydantic_game_state()

    # 2. Summarize memory buffer
    memory_str = "Recent history (last 5 steps):\\n"
    if not memory_buffer:
        memory_str += "No history yet.\\n"
    else:
        for entry in memory_buffer[-5:]:
            action_info = entry.action_taken if entry.action_taken else entry.llm_action_prediction
            reward_info = f"{entry.reward_received:.2f}" if entry.reward_received is not None else "N/A"
            memory_str += f"  Step {entry.step}: Obs (agent @ {entry.observation.agent_pos}, dir {entry.observation.agent_dir}, carrying '{entry.observation.carrying}'), Action: {action_info}, Reward: {reward_info}\\n"
            if entry.llm_thought_process:
                 memory_str += f"    Thought: {entry.llm_thought_process[:100]}...\\n" # Truncate long thoughts

    # 3. Get action from DSPy agent
    with dspy.context(lm=lm):
        prediction = dspy_agent(current_game_state=current_game_state_from_env, memory=memory_str)

    predicted_action_str = prediction.next_action.strip().lower()
    thought = prediction.thought_process

    print(f"LLM Thought: {thought}")
    print(f"LLM Predicted Action: {predicted_action_str}")

    # 4. Store current observation and LLM prediction in memory BEFORE action
    current_memory_entry = MemoryEntry(
        step=current_game_state_from_env.step_count,
        observation=current_game_state_from_env,
        llm_thought_process=thought,
        llm_action_prediction=predicted_action_str
    )
    # This entry will be updated with reward and actual action after step

    # 5. Convert predicted_action_str to game action type and apply to game_env_facade
    # Our MockMiniGridEnv directly takes the string action
    obs, reward, terminated, truncated, info = game_env_facade.step(predicted_action_str)
    print(f"Action taken: {predicted_action_str}, Status: {info.get('last_action_status', 'N/A')}, Reward: {reward:.2f}")


    # 6. Update the current_memory_entry with the actual action taken and reward received,
    # then append to global memory_buffer
    current_memory_entry.action_taken = predicted_action_str # Assuming prediction is directly executed
    current_memory_entry.reward_received = reward
    memory_buffer.append(current_memory_entry)

    # 7. Return new state and outcomes
    return obs, reward, terminated, truncated, info, thought, predicted_action_str


if __name__ == "__main__":
    setup_mlflow()
    lm = configure_dspy()

    # Initialize the agent
    agent = GamePlayingAgent()

    # Initialize mock environment
    game_env = MockMiniGridEnv(max_steps=20) # Limit steps for demo

    print("DSPy environment setup complete. Agent and Mock Environment initialized.")
    print(f"Mission: {game_env.mission}")

    # Game loop
    for i in range(game_env.max_steps):
        print(f"\\n--- Step {game_env.current_step} ---")
        current_state_for_print = game_env.get_pydantic_game_state()
        print(f"Current Grid:\\n{current_state_for_print.grid_representation.replace('WF', 'WW')}") # Make walls more visible
        print(f"Agent @ {current_state_for_print.agent_pos}, Dir: {current_state_for_print.agent_dir}, Carrying: {current_state_for_print.carrying}")

        try:
            obs, reward, terminated, truncated, info, thought, action = run_dspy_agent_turn(
                game_env_facade=game_env,
                dspy_agent=agent,
                lm=lm
            )

            if terminated:
                print(f"\\nEpisode finished after {game_env.current_step} steps: Terminated. Status: {info.get('last_action_status', 'N/A')}")
                break
            if truncated:
                print(f"\\nEpisode finished after {game_env.current_step} steps: Truncated (max steps reached).")
                break

        except Exception as e:
            print(f"Error during agent turn: {e}")
            # print("\\nLM History (last call attempt):")
            # print(lm.inspect_history(n=1)) # Often useful to see what was sent to the LM
            break

    print("\\n--- End of Game ---")
    print(f"Final Grid State:\\n{game_env.get_pydantic_game_state().grid_representation.replace('WF', 'WW')}")
    print(f"Agent @ {game_env.agent_pos}, Dir: {game_env.agent_dir}, Carrying: {game_env.carrying}")
    print(f"Total steps: {game_env.current_step}")

    if memory_buffer:
        print("\\nFinal Memory Buffer (last 10 entries):")
        for entry in memory_buffer[-10:]:
            action_info = entry.action_taken if entry.action_taken else entry.llm_action_prediction
            reward_info = f"{entry.reward_received:.2f}" if entry.reward_received is not None else "N/A"
            print(f"  Step {entry.step}: Obs (agent @ {entry.observation.agent_pos}, dir {entry.observation.agent_dir}, carrying '{entry.observation.carrying}'), Action: {action_info}, Reward: {reward_info}")
            if entry.llm_thought_process:
                 print(f"    Thought: {entry.llm_thought_process}")
