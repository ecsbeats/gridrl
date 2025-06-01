import dspy
import pytest

@pytest.fixture
def lm() -> dspy.LM:
    lm = dspy.LM(model="openai/gemma3:27b",
                api_key="X",
                max_tokens=128000,
                api_base="http://localhost:11434/v1",
                model_type="chat")
    dspy.configure(lm=lm)
    return lm

def test_dspy_hello_world(lm):
    out = lm("Hello, world!")
    print(out)
    assert type(out) == list

