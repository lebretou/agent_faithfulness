# Agent Faithfulness — Internal Signals of Tool-Result Inconsistency

Three-day project investigating whether an open-weights LLM agent (Qwen2.5-7B-Instruct) encodes detectable signals of tool-result inconsistency in its residual stream, even when its chain of thought fails to verbalize them.

See [`project_plan.md`](project_plan.md) for the full backbone.

## Layout (project_plan.md §13)

```
src/        catalog, queries, tool, perturbation, labels, agent, activations, probes, steering
scripts/    01_generate_corpus.py, 02_compute_labels.py, 03_train_probes.py, ...
configs/    default.yaml
tests/      pytest suite for CPU-only modules
notebooks/  day1_smoke.ipynb, day1_overnight.ipynb (run on Colab Pro+ A100)
data/       generated catalog, trajectories.jsonl, activations/  (gitignored)
```

## Local vs Colab

The project splits along a hard line: pure-Python modules (catalog, queries, tool, perturbation, labels) run on a CPU-only Mac; anything model-bound (`agent.py`, `activations.py`, `scripts/01_generate_corpus.py`) runs on Colab Pro+ A100.

### Local dev (Mac)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

### Colab

Open `notebooks/day1_smoke.ipynb`, run cells top to bottom. Then `notebooks/day1_overnight.ipynb` to kick off the 250-trajectory corpus run.

## Day 1 status

- [x] Repo scaffold, configs, gitignore
- [x] `src/schema.py`, `src/catalog.py`, `src/queries.py`, `src/tool.py`
- [x] `src/perturbation.py`, `src/labels.py`
- [x] `src/agent.py`, `src/activations.py`
- [x] `scripts/01_generate_corpus.py`
- [x] `notebooks/day1_smoke.ipynb`, `notebooks/day1_overnight.ipynb`
- [ ] Smoke notebook green on Colab (run by user)
- [ ] Overnight corpus generated (≥200 trajectories on Drive)
