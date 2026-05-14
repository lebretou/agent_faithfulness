# Agent Faithfulness — Internal Signals of Tool-Result Inconsistency

This repository investigates whether an open-weights LLM agent
(`Qwen2.5-7B-Instruct`) encodes detectable signals of tool-result
inconsistency in its residual stream, even when its chain of thought fails to
verbalize them. It contains a controlled synthetic shopping environment, an
agent loop with activation hooks, per-layer linear probes, and an activation
steering sweep.

See [`report/report.pdf`](report/report.pdf) for the full write-up.

## Layout

```
src/        catalog, queries, tool, perturbation, labels, agent, activations, probes, steering
scripts/    01_generate_corpus.py, 02_compute_labels.py, 03_train_probes.py, ...
configs/    default.yaml
tests/      pytest suite for CPU-only modules
notebooks/  smoke, corpus_generation, probes, steering (run on a GPU host)
data/       generated catalog, trajectories.jsonl, activations/  (gitignored)
report/     LaTeX source and compiled PDF
```

## Local vs GPU

The project splits along a hard line: pure-Python modules (catalog, queries,
tool, perturbation, labels) run CPU-only; anything model-bound (`agent.py`,
`activations.py`, `scripts/01_generate_corpus.py`, the steering sweep) needs
a GPU (A100/H100 class).

### Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

### Corpus generation and steering (GPU)

Open `notebooks/smoke.ipynb` to validate the pipeline, then
`notebooks/corpus_generation.ipynb` to produce the trajectory corpus.
`notebooks/probes.ipynb` and `notebooks/steering.ipynb` cover the analysis
stages.
