"""Agent loop with Qwen2.5 chat template + native tool calling.

project_plan.md sections 5.4 and 5.5.

Imports torch/transformers lazily — runs on Colab only.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tool import search
from .perturbation import perturb_tool_result
from .activations import capture_residuals, save_activations, target_position_from_input_ids


SYSTEM_PROMPT = (
    "You are a shopping assistant. Your task is to find an item that matches the user's request. "
    "You have access to a search tool. Reason step by step inside <think>...</think> tags before each action. "
    "After thinking, either call the search tool with refined parameters, or output your final purchase decision "
    "in the format `PURCHASE: item_XXXX`. Do not purchase an item that does not match the user's request."
)

# Qwen2.5 native tool definition for the search tool.
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the catalog for items matching given attribute constraints. "
            "Returns up to k items, ranked by match score (count of satisfied constraints), "
            "ties broken by lower price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "e.g. shirt, pants, shoes, jacket"},
                "color": {"type": "string", "description": "e.g. red, blue, green, black"},
                "size": {"type": "string", "description": "XS, S, M, L, XL"},
                "material": {"type": "string", "description": "cotton, polyester, leather, ..."},
                "brand": {"type": "string"},
                "price_max": {"type": "number", "description": "Maximum price"},
                "k": {"type": "integer", "description": "Number of results to return", "default": 5},
            },
            "required": [],
        },
    },
}


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", dtype: str = "bfloat16"):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tok


# --- Parsing helpers ---------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
_PURCHASE_RE = re.compile(r"PURCHASE:\s*(item_\d+)", re.IGNORECASE)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_assistant_output(text: str) -> dict:
    """Extract thought, optional tool call, optional final purchase from a single assistant turn."""
    thought_match = _THINK_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    purchase_match = _PURCHASE_RE.search(text)
    purchase_id = purchase_match.group(1) if purchase_match else None

    tool_call = None
    tc_match = _TOOLCALL_RE.search(text)
    if tc_match:
        try:
            tool_call = json.loads(tc_match.group(1))
        except json.JSONDecodeError:
            tool_call = None

    return {
        "thought": thought,
        "tool_call": tool_call,
        "purchase_id": purchase_id,
        "raw": text,
    }


def _normalize_tool_args(args: dict) -> tuple[dict, int]:
    """Split a flat search-tool args dict into (constraints, k)."""
    args = dict(args)
    k = int(args.pop("k", 5))
    # Drop None / empty.
    constraints = {kk: vv for kk, vv in args.items() if vv not in (None, "", [])}
    return constraints, k


# --- Main loop ---------------------------------------------------------------

def run_trajectory(
    model,
    tok,
    catalog: list[dict],
    query: dict,
    perturbation_level: int,
    rng: random.Random,
    trajectory_id: str,
    max_steps: int = 5,
    capture_activations: bool = True,
    activations_dir: str | Path | None = None,
    max_new_tokens: int = 512,
) -> dict:
    """Run one agent trajectory end-to-end and return the §5.5 dict."""

    activations_dir = Path(activations_dir) if activations_dir else None

    # Build the initial chat. Qwen2.5's apply_chat_template supports `tools=`.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query["natural_language"]},
    ]

    perturbed_attribute = None
    perturbed_value = None
    original_value = None
    perturbation_applied = False

    steps = []
    final_action = None
    final_item = None

    for step_idx in range(max_steps):
        # Apply chat template; we'll capture activations at the LAST token of the
        # current input (which is the last token of the most recent tool message
        # whenever a tool message was just appended).
        templated = tok.apply_chat_template(
            messages,
            tools=[SEARCH_TOOL_SCHEMA],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Some transformers versions return a BatchEncoding (dict-like) when
        # `tools=` is set; older versions return a raw tensor.
        if isinstance(templated, torch.Tensor):
            prompt_ids = templated.to(model.device)
            attention_mask = torch.ones_like(prompt_ids)
        else:
            prompt_ids = templated["input_ids"].to(model.device)
            attention_mask = templated.get(
                "attention_mask", torch.ones_like(prompt_ids)
            ).to(model.device)

        # Activation capture: only meaningful AFTER a tool message has been appended,
        # i.e., on steps where the previous step issued a tool call.
        activations_path = None
        last_msg = messages[-1]
        if (
            capture_activations
            and last_msg.get("role") == "tool"
            and activations_dir is not None
        ):
            target = target_position_from_input_ids(prompt_ids)
            acts = capture_residuals(model, prompt_ids, target)
            activations_path = str(
                activations_dir / f"{trajectory_id}_step_{step_idx}.pt"
            )
            save_activations(acts, activations_path)

        # Generate one assistant turn.
        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_ids = out_ids[0, prompt_ids.shape[-1]:]
        text = tok.decode(new_ids, skip_special_tokens=False)

        parsed = parse_assistant_output(text)

        step_record = {
            "step_idx": step_idx,
            "thought": parsed["thought"],
            "tool_call": None,
            "tool_result": None,
            "tool_result_clean": None,
            "activations_path": activations_path,
            "raw_assistant": parsed["raw"],
        }

        # Final purchase?
        if parsed["purchase_id"] is not None:
            steps.append(step_record)
            final_action = f"PURCHASE: {parsed['purchase_id']}"
            final_item = next(
                (it for it in catalog if it["item_id"] == parsed["purchase_id"]), None
            )
            messages.append({"role": "assistant", "content": parsed["raw"]})
            break

        # Tool call?
        if parsed["tool_call"] is not None:
            args = parsed["tool_call"].get("arguments", parsed["tool_call"])
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            constraints, k = _normalize_tool_args(args)
            clean_result = search(catalog, {"constraints": constraints}, k=k)

            if not perturbation_applied and perturbation_level != 0:
                perturbed_result, p_attr, p_vals = perturb_tool_result(
                    clean_result, perturbation_level, query["constraints"], rng
                )
                if p_attr is not None:
                    perturbation_applied = True
                    perturbed_attribute = p_attr
                    original_value, perturbed_value = p_vals
            else:
                perturbed_result = clean_result

            step_record["tool_call"] = {"arguments": {**constraints, "k": k}}
            step_record["tool_result"] = perturbed_result
            step_record["tool_result_clean"] = clean_result

            messages.append({"role": "assistant", "content": parsed["raw"]})
            messages.append(
                {"role": "tool", "name": "search", "content": json.dumps(perturbed_result)}
            )
            steps.append(step_record)
            continue

        # Neither — assistant didn't follow the protocol. Record and break.
        steps.append(step_record)
        messages.append({"role": "assistant", "content": parsed["raw"]})
        break

    trajectory = {
        "trajectory_id": trajectory_id,
        "query": query,
        "perturbation_level": perturbation_level,
        "perturbed_attribute": perturbed_attribute,
        "original_value": original_value,
        "perturbed_value": perturbed_value,
        "steps": steps,
        "final_action": final_action,
        "final_item": final_item,
        "labels": {},
    }
    return trajectory
