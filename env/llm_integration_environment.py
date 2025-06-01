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


# Example usage (conceptual, needs to be integrated with the game loop)
def run_dspy_agent_turn(
    game_env_facade: Any, # A facade or wrapper around your actual MiniGrid env
    dspy_agent: GamePlayingAgent,
    lm: dspy.LM
):
    # 1. Get current game state from game_env_facade and convert to GameState Pydantic model
    # This function would need to be implemented to extract info from your MiniGrid env
    # and its pprint_grid output.
    current_game_state_from_env = game_env_facade.get_pydantic_game_state() # Placeholder

    # 2. Summarize memory buffer (e.g., last 5 entries)
    # For simplicity, let's create a string summary for now.
    memory_str = "Recent history (last 5 steps):\n"
    if not memory_buffer:
        memory_str += "No history yet.\n"
    else:
        for entry in memory_buffer[-5:]:
            memory_str += f"  Step {entry.step}: Obs (agent @ {entry.observation.agent_pos}), Action: {entry.action_taken or entry.llm_action_prediction}, Reward: {entry.reward_received}\n"

    # 3. Get action from DSPy agent
    with dspy.context(lm=lm):
        prediction = dspy_agent(current_game_state=current_game_state_from_env, memory=memory_str)

    predicted_action_str = prediction.next_action.strip().lower()
    thought = prediction.thought_process

    print(f"LLM Thought: {thought}")
    print(f"LLM Predicted Action: {predicted_action_str}")

    # 4. Store current observation and LLM prediction in memory
    current_memory_entry = MemoryEntry(
        step=current_game_state_from_env.step_count,
        observation=current_game_state_from_env,
        llm_thought_process=thought,
        llm_action_prediction=predicted_action_str
    )
    # We will append this to memory_buffer after the action is taken and reward is known

    # 5. TODO: Convert predicted_action_str to game action type and apply to game_env_facade
    # game_action = convert_str_to_game_action(predicted_action_str) # Placeholder
    # obs, reward, terminated, truncated, info = game_env_facade.step(game_action) # Placeholder

    # 6. TODO: Update the current_memory_entry with the actual action taken and reward received,
    # then append to global memory_buffer
    # current_memory_entry.action_taken = actual_action_str (if different or validated)
    # current_memory_entry.reward_received = reward
    # memory_buffer.append(current_memory_entry)

    # 7. TODO: Handle terminated/truncated, render, etc.

    return predicted_action_str, thought

if __name__ == "__main__":
    setup_mlflow()
    lm = configure_dspy()

    # Initialize the agent
    agent = GamePlayingAgent()

    # This is a conceptual example. We need a game environment loop.
    print("DSPy environment setup complete. Agent initialized.")
    print("Next steps would involve creating a game loop that uses `run_dspy_agent_turn`.")

    # Simple test of the signature with dummy data:
    dummy_grid = """\
WFWFWFWFWF
WF>>    WF
WF      WF
WF  KB  WF
WF      WF
WFWFWFWFWF""".strip()
    dummy_state = GameState(
        grid_representation=dummy_grid,
        agent_pos=(1,1), # Corresponds to >> in the grid if 0-indexed
        agent_dir=0, # Facing right
        mission="Find the green key (KB) and toggle it.",
        carrying=None,
        step_count=1,
        max_steps=100
        # success_pos and failure_pos are no longer part of GameState
    )
    dummy_memory = "No history yet."

    try:
        with dspy.context(lm=lm):
            print("\nTesting agent prediction with dummy state...")
            prediction = agent(current_game_state=dummy_state, memory=dummy_memory)
            print(f"Predicted action: {prediction.next_action}")
            print(f"Thought process: {prediction.thought_process}")

        # print("\nLM History (last 1 call):")
        # print(lm.inspect_history(n=1))

    except Exception as e:
        print(f"Error during dummy agent prediction: {e}")
        # print("\nLM History (last call attempt):")
        # print(lm.inspect_history(n=1)) # Often useful to see what was sent to the LM

    # qa = dspy.ChainOfThought("question -> answer")
    # response = qa(question="What is the capital of France?")
    # print(response.answer)
