from src.labels import (
    cot_mentions_perturbation,
    action_rejects_perturbation,
    task_success,
    compute_all_labels,
)


def test_verbalization_catches_value_mention():
    assert cot_mentions_perturbation(["The result shows a red shirt."], "red", "blue")
    assert not cot_mentions_perturbation(["The result shows a shirt."], "red", "blue")


def test_verbalization_catches_acknowledgment_phrase():
    assert cot_mentions_perturbation(
        ["This doesn't match what was asked."], "x", "y"
    )
    assert cot_mentions_perturbation(
        ["The brand is different from the request."], "BrandX", "BrandY"
    )


def test_verbalization_handles_string_inputs():
    assert cot_mentions_perturbation("The color is wrong.", "red", "blue")
    assert not cot_mentions_perturbation("", "red", "blue")


def test_action_rejects_when_purchase_differs():
    assert action_rejects_perturbation("item_1", "item_0", 0)
    assert not action_rejects_perturbation("item_0", "item_0", 0)


def test_action_rejects_when_many_searches():
    assert action_rejects_perturbation("item_0", "item_0", n_search_calls_after_perturbation=2)


def test_task_success_constraint_check():
    item = {"item_id": "x", "category": "shirt", "color": "blue", "price": 20.0,
            "size": "M", "material": "cotton", "brand": "BrandA"}
    assert task_success(item, {"category": "shirt", "color": "blue"})
    assert not task_success(item, {"color": "red"})
    assert not task_success(None, {"color": "red"})


def test_compute_all_labels_on_fixture():
    traj = {
        "query": {"constraints": {"category": "shirt", "color": "blue"}},
        "perturbation_level": 2,
        "perturbed_attribute": "color",
        "original_value": "blue",
        "perturbed_value": "red",
        "steps": [
            {
                "step_idx": 0,
                "thought": "I see a red shirt — that doesn't match the request.",
                "tool_call": {"query": {"category": "shirt", "color": "blue"}, "k": 5},
                "tool_result": [{"item_id": "item_0", "category": "shirt", "color": "red",
                                 "size": "M", "price": 20.0, "material": "cotton",
                                 "brand": "BrandA"}],
                "tool_result_clean": [{"item_id": "item_0", "category": "shirt",
                                        "color": "blue", "size": "M", "price": 20.0,
                                        "material": "cotton", "brand": "BrandA"}],
            },
            {
                "step_idx": 1,
                "thought": "Let me purchase the second-best option.",
                "tool_call": None,
                "tool_result": None,
                "tool_result_clean": None,
            },
        ],
        "final_item": {"item_id": "item_5", "category": "shirt", "color": "blue",
                       "size": "M", "price": 25.0, "material": "cotton",
                       "brand": "BrandX"},
    }
    labels = compute_all_labels(traj)
    assert labels["cot_mentions_perturbation"] is True
    assert labels["action_rejects_perturbation"] is True
    assert labels["task_success"] is True
