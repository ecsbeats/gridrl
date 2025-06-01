import dspy
import mlflow
from typing import List, Tuple, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum
from memory import MemoryEnv
import os
import gymnasium
from dotenv import load_dotenv
from gymnasium.wrappers import RecordVideo
from dspy import Evaluate, Example
import json
import tempfile # Added for temporary video storage
import glob     # Added for finding video files
import shutil   # Added for cleaning up video folders if necessary
import traceback # Added for detailed exception logging

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

def get_game_state(env: gymnasium.Env) -> GameState:
    # Access the underlying environment if wrapped
    current_env = env
    while hasattr(current_env, 'env') and not hasattr(current_env, 'pprint_grid'):
        current_env = current_env.env

    unwrapped_env = current_env

    return GameState(
        grid_representation=unwrapped_env.pprint_grid(),
        agent_pos=unwrapped_env.agent_pos,
        agent_dir=unwrapped_env.agent_dir,
        mission=unwrapped_env.mission,
        carrying=unwrapped_env.carrying,
        step_count=unwrapped_env.step_count,
        max_steps=unwrapped_env.max_steps,
        success_pos=unwrapped_env.success_pos,
        failure_pos=unwrapped_env.failure_pos,
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
    action: Literal['forward', 'left', 'right', 'toggle', 'pickup', 'drop'] = dspy.OutputField(description="Action can be one of the following: left (rotates left), right (rotates right), forward (moves forward in the direction you are facing), toggle (toggles a door), pickup, drop") # not including done due to early stopping

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
        super().__init__()
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

    def reset_agent_state(self):
        """Resets the internal state of the agent for a new episode."""
        self.short_term_memory = []
        self.long_term_memory = [] # Or decide if LTM should persist across eval episodes
        self.steps = 0
        self.last_game_state = None
        self.last_action = None
        self.last_analysis = ""
        self.scratchpad = ""
        self.last_reward = 0
        self.last_feedback = ""

    def forward(self, env: MemoryEnv):
        game_state = get_game_state(env)

        self.last_game_state = game_state
        current_previous_action = self.last_action if self.last_action else "None (first step)"
        current_feedback = self.last_feedback if self.last_feedback else "None (first step)"

        generate_actions_output = self.generate_action(game_state=game_state, previous_action=current_previous_action, scratchpad=self.scratchpad, feedback=current_feedback)
        self.last_action = generate_actions_output.action

        self.steps += 1
        obs, reward, terminated, truncated, info = env.step(string_to_action(self.last_action))

        new_game_state = get_game_state(env)
        analysis = self.generate_mechanics_analysis(old_game_state=self.last_game_state, action=self.last_action, new_game_state=new_game_state)
        self.scratchpad += analysis.new_notes_for_scratchpad + "\n"
        self.last_analysis = analysis.game_mechanics_analysis
        self.last_feedback = self.last_analysis
        self.last_reward = reward

        return self.last_action, obs, reward, terminated, truncated, info

# New Class for making GamePlayingAgent evaluatable by dspy.Evaluate
class EvaluatableAgent(dspy.Module):
    def __init__(self, agent: GamePlayingAgent):
        super().__init__()
        self.agent = agent
        self.record_videos = os.getenv("EVAL_VIDEO_RECORDING", "False").lower() == "true"

    def __call__(self, game_config: dict) -> dict:
        try:
            env_size = game_config.get('size', 13)
            env_random_length = game_config.get('random_length', False)
            env_max_steps = game_config.get('max_steps', 100)
            env_seed = game_config.get('seed')

            video_artifact_path = None
            temp_video_dir_obj = None # Ensure it's defined for finally block
            temp_video_dir = None

            current_render_mode = 'rgb_array' if self.record_videos else 'text'

            env = MemoryEnv(
                size=env_size,
                random_length=env_random_length,
                max_steps=env_max_steps,
                render_mode=current_render_mode
            )

            if self.record_videos:
                try:
                    temp_video_dir_obj = tempfile.TemporaryDirectory() # Assign here
                    temp_video_dir = temp_video_dir_obj.name
                    video_name_prefix = f"eval_s{env_size}_seed{env_seed}_id{game_config.get('example_id', 'N_A')}"
                    env = RecordVideo(
                        env,
                        video_folder=temp_video_dir,
                        name_prefix=video_name_prefix,
                        episode_trigger=lambda x: True
                    )
                except Exception as e_video_setup:
                    print(f"Warning: Could not set up video recording for game_config {game_config}: {e_video_setup}.")
                    traceback.print_exc() # Print traceback for video setup error
                    # Fallback to no video recording for this specific call
                    current_record_videos_for_this_call = False
                    # Re-initialize env with text mode if RecordVideo setup failed after MemoryEnv init
                    if current_render_mode == 'rgb_array':
                        env.close()
                        env = MemoryEnv(size=env_size, random_length=env_random_length, max_steps=env_max_steps, render_mode='text')
            else:
                current_record_videos_for_this_call = False

            obs, info = env.reset(seed=env_seed)
            self.agent.reset_agent_state()

            terminated = False
            truncated = False
            total_reward_for_episode = 0.0
            steps_taken = 0

            while not terminated and not truncated:
                action_str, obs, reward, terminated, truncated, info = self.agent.forward(env)
                total_reward_for_episode += reward
                steps_taken += 1

            env.close()

            if self.record_videos and temp_video_dir: # Check self.record_videos (original intent) and if dir was created
                try:
                    video_files = glob.glob(os.path.join(temp_video_dir, f"{video_name_prefix}-episode-*.mp4"))
                    if video_files:
                        for video_path_on_disk in video_files:
                            artifact_video_name = os.path.basename(video_path_on_disk)
                            mlflow.log_artifact(video_path_on_disk, artifact_path=f"evaluation_videos/{artifact_video_name}")
                            print(f"Logged video artifact: evaluation_videos/{artifact_video_name}")
                            video_artifact_path = f"evaluation_videos/{artifact_video_name}"
                    else:
                        print(f"Warning: No video file found for prefix {video_name_prefix} in {temp_video_dir}")
                except Exception as e_video_log:
                    print(f"Error logging video artifact for game_config {game_config}: {e_video_log}")
                    traceback.print_exc()
                finally:
                    if temp_video_dir_obj:
                        temp_video_dir_obj.cleanup()

            success = terminated and total_reward_for_episode > 0
            return {
                "mission_accomplished": success,
                "total_reward": total_reward_for_episode,
                "steps_taken": steps_taken,
                "game_config_used": game_config,
                "video_artifact_path": video_artifact_path
            }
        except Exception as e_main_call:
            print(f"CRITICAL ERROR in EvaluatableAgent.__call__ for game_config {game_config}:")
            traceback.print_exc() # Print the full traceback of the original error
            # Propagate the error so dspy's parallelizer knows something went wrong
            # Or, return a specific error structure if preferred by dspy.Evaluate error handling
            raise e_main_call

class Game:
    def __init__(self, size=13, random_length=False, max_steps=100, render_mode: str ="text", video_folder: str = "videos"):
        self.render_mode = render_mode
        env_render_mode = "rgb_array" if render_mode == "video" else render_mode

        self.env = MemoryEnv(
            size=size,
            random_length=random_length,
            max_steps=max_steps,
            render_mode=env_render_mode
        )

        if render_mode == "video":
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            self.env = RecordVideo(
                self.env,
                video_folder=video_folder,
                episode_trigger=lambda x: True,
                name_prefix=f"memory-env-s{size}-rl{str(random_length).lower()}"
            )
        self.agent = GamePlayingAgent()

    def run(self):
        self.agent.reset_agent_state() # Reset agent before a standard game run too
        self.env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            if self.render_mode == "human":
                self.env.render()
            action, obs, reward, terminated, truncated, info = self.agent.forward(self.env)
        print(f"Episode finished. Total reward: {self.agent.last_reward}, Steps: {self.agent.steps}")
        self.env.close()

# New function to generate the development set for evaluation
def generate_dev_set(num_examples: int = 10, sizes: List[int] = [8], max_steps_map: Optional[dict[int, int]] = None) -> List[dspy.Example]:
    dev_set = []
    if max_steps_map is None:
        # Default max steps: 5 * size for a tighter challenge based on MemoryEnv defaults for smaller sizes
        max_steps_map = {8: 5 * 8, 11: 5 * 11, 13: 5*13} # Example: 5*size or more specific values

    for i in range(num_examples):
        size = sizes[i % len(sizes)]
        seed = i # Use index as seed for reproducibility, ensures different envs
        current_max_steps = max_steps_map.get(size, 5 * size**2) # Fallback to MemoryEnv default if size not in map

        game_config = {
            'size': size,
            'random_length': False, # Keep consistent for evaluation
            'max_steps': current_max_steps,
            'seed': seed,
            'example_id': i # Add example_id for unique video naming
        }
        # The 'expected_success' field is a convention for dspy.Example if you have labels.
        # Our metric will determine if the run was successful based on the agent's output.
        example = dspy.Example(game_config=game_config, expected_success=True).with_inputs('game_config')
        dev_set.append(example)
    return dev_set

# New metric function for dspy.Evaluate
def eval_metric(gold: dspy.Example, pred: dict, trace=None) -> bool:
    """
    Compares the prediction (output of EvaluatableAgent) against the gold standard (Example).
    Returns True if the agent succeeded based on the prediction, False otherwise.
    """
    # The 'gold' example might contain an 'expected_success' field, but for now,
    # we purely rely on the 'mission_accomplished' field from the prediction (agent's run output).
    return pred.get('mission_accomplished', False)

if __name__ == "__main__":
    setup_mlflow()
    configure_dspy()

    evaluation_mode = os.getenv("EVALUATION", "False").lower() == "true"
    eval_video_recording = os.getenv("EVAL_VIDEO_RECORDING", "False").lower() == "true"

    if evaluation_mode:
        print("Running in EVALUATION mode.")
        if eval_video_recording:
            print("Evaluation video recording is ENABLED.")
        else:
            print("Evaluation video recording is DISABLED.")

        with mlflow.start_run(run_name="dspy_evaluation_run") as run:
            mlflow.log_param("evaluation_mode", True)
            mlflow.log_param("evaluation_video_recording", eval_video_recording)

            game_playing_agent = GamePlayingAgent()
            # Pass eval_video_recording to EvaluatableAgent if it needs to know during init
            # Or, EvaluatableAgent can read env var itself, as currently implemented.
            evaluatable_agent = EvaluatableAgent(agent=game_playing_agent)

            # Configure dev set generation from environment variables
            num_eval_examples = int(os.getenv("NUM_EVAL_EXAMPLES", "10"))
            eval_game_sizes_str = os.getenv("EVAL_GAME_SIZES", "13,17")
            eval_game_sizes = [int(s.strip()) for s in eval_game_sizes_str.split(',') if s.strip()]
            max_steps_per_size_str = os.getenv("EVAL_MAX_STEPS_MAP", "8:40,11:55") # e.g. "8:40,11:55"
            max_steps_map_eval = None # Initialize to ensure it's defined
            if max_steps_per_size_str:
                try:
                    max_steps_map_eval = dict(item.split(':') for item in max_steps_per_size_str.split(','))
                    max_steps_map_eval = {int(k): int(v) for k, v in max_steps_map_eval.items()}
                except ValueError:
                    print(f"Warning: Could not parse EVAL_MAX_STEPS_MAP: '{max_steps_per_size_str}'. Using defaults.")
                    max_steps_map_eval = None # Fallback to default in generate_dev_set

            mlflow.log_params({
                "num_eval_examples": num_eval_examples,
                "eval_game_sizes": eval_game_sizes_str,
                "eval_max_steps_map_config": max_steps_per_size_str, # Log the raw string config
                "eval_max_steps_map_used": str(max_steps_map_eval if max_steps_map_eval else generate_dev_set(0,[],{})._default_max_steps)
            })

            print(f"Generating {num_eval_examples} evaluation examples for sizes {eval_game_sizes} with max steps map {max_steps_map_eval}...")
            dev_set = generate_dev_set(num_examples=num_eval_examples, sizes=eval_game_sizes, max_steps_map=max_steps_map_eval)

            # Save dev_set as an MLflow artifact
            try:
                # dspy.Example is not directly JSON serializable. Convert to dicts.
                serializable_devset = [ex.toDict() for ex in dev_set]
                devset_path = "evaluation_devset.json"
                with open(devset_path, "w") as f:
                    json.dump(serializable_devset, f, indent=2)
                mlflow.log_artifact(devset_path, "evaluation_artifacts")
                print(f"Logged devset to {devset_path} and as MLflow artifact.")
            except Exception as e:
                print(f"Error serializing or logging devset: {e}")

            print("Starting evaluation...")
            eval_num_threads = 1
            try:
                eval_num_threads = int(os.getenv("EVAL_NUM_THREADS", "1"))
                if eval_num_threads <= 0:
                    eval_num_threads = 1
                    print("Warning: EVAL_NUM_THREADS was <= 0, defaulting to 1.")
            except ValueError:
                print("Warning: EVAL_NUM_THREADS environment variable is not a valid integer. Defaulting to 1.")
                eval_num_threads = 1

            mlflow.log_param("evaluation_num_threads", eval_num_threads)
            print(f"Using {eval_num_threads} threads for evaluation.")

            evaluator = Evaluate(devset=dev_set, metric=eval_metric, num_threads=eval_num_threads, display_progress=True, display_table=min(5, num_eval_examples))
            results = evaluator(evaluatable_agent) # `results` is the average score (success rate here)

            print(f"Evaluation finished. Average success rate: {results:.4f}")
            mlflow.log_metric("average_success_rate", results)

            # Log the DSPy program (EvaluatableAgent)
            print("Logging DSPy program (EvaluatableAgent) to MLflow...")
            try:
                # For log_model to work well, especially for loading back,
                # the module should be self-contained or dependencies handled.
                # Providing an input_example helps schema inference.
                if dev_set:
                    input_example_for_log = dev_set[0].without('expected_success')
                else:
                    # Create a dummy example if dev_set is empty (though unlikely here)
                    dummy_game_config = {'size': 8, 'random_length': False, 'max_steps': 40, 'seed': 0}
                    input_example_for_log = dspy.Example(game_config=dummy_game_config).with_inputs('game_config')

                mlflow.dspy.log_model(
                    evaluatable_agent,
                    artifact_path="evaluatable_agent_program",
                    input_example=input_example_for_log
                )
                print("DSPy program (EvaluatableAgent) logged.")
            except Exception as e:
                print(f"Error logging EvaluatableAgent DSPy model: {e}")

            print(f"MLflow Run ID: {run.info.run_id}")
            print("Evaluation complete. Check MLflow UI for results.")

    else:
        print("Running in standard GAME mode.")
        with mlflow.start_run(run_name="dspy_game_run") as run:
            mlflow.log_param("evaluation_mode", False)

            render_mode_env = os.getenv("RENDER_MODE", "text")
            game_size_env = int(os.getenv("GAME_SIZE", "13"))
            random_length_env = os.getenv("GAME_RANDOM_LENGTH", "False").lower() == "true"
            # GAME_MAX_STEPS default should be reasonable, e.g., 5 * size^2 from MemoryEnv
            default_max_steps = 5 * game_size_env * game_size_env
            max_steps_env = int(os.getenv("GAME_MAX_STEPS", str(default_max_steps)))
            video_folder_env = os.getenv("VIDEO_FOLDER", "videos")

            mlflow.log_params({
                "render_mode": render_mode_env,
                "game_size": game_size_env,
                "random_length": random_length_env,
                "max_steps_game": max_steps_env,
                "video_folder": video_folder_env
            })

            game = Game(
                render_mode=render_mode_env,
                size=game_size_env,
                random_length=random_length_env,
                max_steps=max_steps_env,
                video_folder=video_folder_env
            )
            game.run()

            # Log the GamePlayingAgent after a standard run
            print("Logging GamePlayingAgent to MLflow...")
            try:
                # Attempt to create a representative input example for GamePlayingAgent
                # This requires an active environment to get a GameState
                # For simplicity, if direct logging fails, consider logging key parameters manually
                # or ensure the agent can be initialized without a live env for signature purposes.

                # Create a temporary env to get a sample GameState for input_example
                temp_env_for_signature = MemoryEnv(size=game_size_env, max_steps=10, render_mode='text')
                temp_env_for_signature.reset()
                sample_game_state = get_game_state(temp_env_for_signature)
                temp_env_for_signature.close()

                input_example_for_game_agent = dspy.Example(
                    game_state=sample_game_state,
                    previous_action="None",
                    scratchpad="Initial scratchpad",
                    feedback="None"
                ).with_inputs('game_state', 'previous_action', 'scratchpad', 'feedback')

                mlflow.dspy.log_model(
                    game.agent, # The GamePlayingAgent instance
                    artifact_path="game_playing_agent_program",
                    input_example=input_example_for_game_agent
                )
                print("GamePlayingAgent logged.")
            except Exception as e:
                print(f"Error logging GamePlayingAgent DSPy model: {e}")

            print(f"MLflow Run ID: {run.info.run_id}")
            print("Game run complete. Check MLflow UI.")

# Helper to get default max_steps for logging in case max_steps_map_eval is None
# This is a bit of a hack; ideally, generate_dev_set would expose its defaults more cleanly.
# For now, ensure _default_max_steps is available if generate_dev_set is called with 0 examples
# (as in the MLflow logging line if max_steps_map_eval is None)
if not hasattr(generate_dev_set, '_default_max_steps'):
    generate_dev_set._default_max_steps = {8: 5 * 8, 11: 5 * 11, 13: 5*13}
