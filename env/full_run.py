import dspy
import mlflow
from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum
from memory import MemoryEnv
import os
import gymnasium
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    mlflow.dspy.autolog()
    mlflow.set_experiment("dspy-full-run")

def configure_dspy() -> dspy.LM:
    lm = get_policy_lm()
    dspy.configure(lm=lm)
    return lm

def get_policy_lm() -> dspy.LM:
    lm = dspy.LM(model=os.getenv("MODEL_NAME", "openai/gemma3:27b"),
                api_key=os.getenv("MODEL_API_KEY", "X"),
                max_tokens=128000,
                api_base=os.getenv("DSPY_API_BASE", "http://localhost:11434/v1"),
                model_type=os.getenv("MODEL_TYPE", "chat"))
    return lm

class ShortTermMemoryEntry(BaseModel):
    step: int
    scratchpad: str = Field(..., description="The scratchpad/memory of the agent.")
    reward_received: Optional[float] = Field(None, description="The reward received by the agent.")
    action_taken: Optional[str] = Field(None, description="The action taken by the agent.")
    llm_thought_process: Optional[str] = Field(None, description="The thought process of the agent.")

class LongTermMemoryEntry(BaseModel):
    learnings: Optional[str] = Field(None, description="A summary of the learnings from this episode.")

class GameState(BaseModel):
    grid_representation: str = Field(..., description="""
                                    ASCII string representation of the full game grid.
                                    The grid is a 2D array of characters (grouped in 2s),
                                    The first character is the object in the cell, the second character is the color of the object.
                                    Objects include:
                                    Wall: W
                                    Floor: F
                                    Door: D
                                    Key: K
                                    Ball: A
                                    Box: B
                                    Goal: G
                                    Lava: V
                                    Colors include:
                                    Red: A
                                    Green: B
                                    Blue: C
                                    Purple: D
                                    Yellow: E
                                    Grey: F
                                    The grid is 2D, so the first character is the object in the cell, the second character is the color of the object.
                                    Your position is given by >> (facing right), VV (facing down), << (facing left), ^^ (facing up).
                                    """)
    agent_pos: Tuple[int, int] = Field(..., description="Agent's current (x, y) position.")
    agent_dir: int = Field(..., description="Agent's current direction (0: right, 1: down, 2: left, 3: up).")
    mission: str = Field(..., description="The mission for the agent.")
    carrying: Optional[str] = Field(None, description="Description of the object the agent is carrying, if any (e.g., 'green key').")
    step_count: int = Field(..., description="Current step count in the episode.")
    max_steps: int = Field(..., description="Maximum steps allowed in the episode.")
    success_pos: Tuple[int, int] = Field(..., description="The position of the success object.")
    failure_pos: Tuple[int, int] = Field(..., description="The position of the failure object.")
    short_term_memory: List[ShortTermMemoryEntry] = Field(..., description="The short-term memory of the agent.")
    long_term_memory: List[LongTermMemoryEntry] = Field(..., description="The long-term memory of the agent.")

string_to_action_map = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}

def string_to_action(action: str) -> int:
    return string_to_action_map[action]

def get_game_state(env: MemoryEnv) -> GameState:
    return GameState(
        grid_representation=env.pprint_grid(),
        agent_pos=env.agent_pos,
        agent_dir=env.agent_dir,
        mission=env.mission,
        carrying=env.carrying,
        step_count=env.step_count,
        max_steps=env.max_steps,
        success_pos=env.success_pos,
        failure_pos=env.failure_pos,
        short_term_memory=[], # TODO: add short term memory
        long_term_memory=[], # TODO: add long term memory
    )

class GenerateActionSignature(dspy.Signature):
    """
    Generate an action for the agent to take.
    """

    game_state: GameState = dspy.InputField(description="The current game state.")
    scratchpad: str = dspy.InputField(description="Your scratchpad/memory of the game.")
    previous_action: str = dspy.InputField(description="The previous action you took.")
    feedback: str = dspy.InputField(description="Feedback on the previous action you took from your friend.")
    action: str = dspy.OutputField(description="Action can be one of the following: left (rotates left), right (rotates right), forward (moves forward in the direction you are facing), toggle (toggles a door), pickup, drop") # not including done due to early stopping

class GenerateMechanicsAnalysisSignature(dspy.Signature):
    """
    Generate an analysis of the game mechanics and add notes to the scratchpad for you to remember.
    """

    old_game_state: GameState = dspy.InputField(description="The previous game state.")
    action: str = dspy.InputField(description="Your last action.")
    new_game_state: GameState = dspy.InputField(description="The new game state.")
    game_mechanics_analysis: str = dspy.OutputField(description="A summary of the game mechanics and the action taken.")
    new_notes_for_scratchpad: str = dspy.OutputField(description="Notes to add to your scratchpad/memory. Please note when something is not working as expected (wrong orientation, wrong direction after action, etc.)")

class GamePlayingAgent(dspy.Module):
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = []
        self.steps = 0
        self.last_game_state = None
        self.last_action = None
        self.last_analysis = ""
        self.scratchpad = ""
        self.last_reward = 0
        self.last_feedback = ""

        self.generate_action = dspy.ChainOfThought(GenerateActionSignature)
        self.generate_mechanics_analysis = dspy.ChainOfThought(GenerateMechanicsAnalysisSignature)

    def update_short_term_memory(self, entry: ShortTermMemoryEntry):
        self.short_term_memory.append(entry)

    def update_long_term_memory(self, entry: LongTermMemoryEntry):
        self.long_term_memory.append(entry)

    def forward(self, env: MemoryEnv):
        game_state = get_game_state(env)

        self.last_game_state = game_state
        generate_actions_output = self.generate_action(game_state=game_state, previous_action=self.last_action, scratchpad=self.scratchpad, feedback=self.last_feedback)
        self.last_action = generate_actions_output.action

        self.steps += 1
        obs, reward, terminated, truncated, info = env.step(string_to_action(self.last_action))

        new_game_state = get_game_state(env)
        analysis = self.generate_mechanics_analysis(old_game_state=self.last_game_state, action=self.last_action, new_game_state=new_game_state)
        self.scratchpad += analysis.new_notes_for_scratchpad
        self.last_analysis = analysis.game_mechanics_analysis
        self.last_feedback = self.last_analysis
        self.last_reward = reward

        return self.last_action, obs, reward, terminated, truncated, info



class Game:
    def __init__(self, size=13, random_length=False, max_steps=100, render_mode="text"):
        self.env = MemoryEnv(
            size=size,
            random_length=False,
            max_steps=100,
            render_mode=render_mode
        )
        self.agent = GamePlayingAgent()

    def run(self):
        self.env.reset()
        terminated = False
        while not terminated:
            if self.env.render_mode == "human":
                self.env.render()
            action, obs, reward, terminated, truncated, info = self.agent.forward(self.env)
        print(f"Total reward: {self.agent.last_reward}")

if __name__ == "__main__":
    setup_mlflow()
    configure_dspy()
    with mlflow.start_run():
        game = Game(render_mode=os.getenv("RENDER_MODE", "text"), size=int(os.getenv("GAME_SIZE", 13)), random_length=os.getenv("GAME_RANDOM_LENGTH", False))
        game.run()
