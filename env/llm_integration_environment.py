import dspy
import mlflow

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

if __name__ == "__main__":
    setup_mlflow()
    lm = configure_dspy()
    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question="What is the capital of France?")
    print(response.answer)
