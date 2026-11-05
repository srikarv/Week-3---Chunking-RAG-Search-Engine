import json
from pathlib import Path
import pytest

from test_adapter import ask_chatbot

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "test_cases.json").read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.get("id", "case"))
def test_chatbot_returns_nonempty_answer(case):
    answer = ask_chatbot(case["question"])
    assert isinstance(answer, str)
    assert answer.strip(), "The chatbot returned an empty answer."


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.get("type") == "supported"],
    ids=lambda c: c.get("id", "supported"),
)
def test_supported_answers_include_expected_concepts(case):
    """Simple requirements-based oracle using expected concepts/keywords.

    This is intentionally NOT exact-string matching because LLM wording is stochastic.
    Replace or improve this oracle if your project has a better ground-truth mechanism.
    """
    answer = normalize(ask_chatbot(case["question"]))
    keywords = [normalize(k) for k in case.get("expected_keywords", []) if k.strip()]
    assert keywords, f"Add human-verified expected_keywords for {case['id']}"
    assert any(k in answer for k in keywords), (
        f"Expected at least one verified concept {keywords}, but got: {answer}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.get("type") == "unsupported"],
    ids=lambda c: c.get("id", "unsupported"),
)
def test_unsupported_question_is_handled_conservatively(case):
    """Heuristic check only; human review is still required.

    Customize the phrases to match the behavior you intentionally designed.
    """
    answer = normalize(ask_chatbot(case["question"]))
    conservative_markers = [
        "not in the provided",
        "not available",
        "cannot determine",
        "don't have enough information",
        "do not have enough information",
        "not found",
        "based on the provided documents",
    ]
    assert any(marker in answer for marker in conservative_markers), (
        "The answer did not contain an obvious uncertainty/out-of-scope marker. "
        "Inspect it manually before deciding whether the chatbot hallucinated.\n"
        f"Answer: {answer}"
    )
