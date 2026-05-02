"""Tests for the assistant-output parser. Pure-Python — no model needed."""
from src.agent import parse_assistant_output


def test_thought_extracted_from_think_tags():
    raw = "<think>I need a blue shirt.</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"color\": \"blue\"}}\n</tool_call>"
    p = parse_assistant_output(raw)
    assert p["thought"] == "I need a blue shirt."
    assert p["tool_call"] == {"name": "search", "arguments": {"color": "blue"}}


def test_thought_falls_back_when_no_think_tags():
    raw = (
        "The closest match was priced at $87.1, which exceeds the budget.\n"
        "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call><|im_end|>"
    )
    p = parse_assistant_output(raw)
    assert "$87.1" in p["thought"]
    assert "exceeds the budget" in p["thought"]
    assert "<tool_call>" not in p["thought"]
    assert "<|im_end|>" not in p["thought"]


def test_purchase_parsed():
    raw = "PURCHASE: item_0042<|im_end|>"
    p = parse_assistant_output(raw)
    assert p["purchase_id"] == "item_0042"


def test_thought_fallback_keeps_purchase_text():
    raw = "Found it: PURCHASE: item_0007<|im_end|>"
    p = parse_assistant_output(raw)
    assert p["purchase_id"] == "item_0007"
    assert "PURCHASE" in p["thought"]
