# TinyGPT — TODO List

This file tracks **all tasks for building, experimenting, and analyzing TinyGPT**.  
Tasks are categorized and prioritized for research and implementation rigor.

---

## Phase 0 — Setup & Infrastructure

- [ X ] Create Python virtual environment / `requirements.txt`
- [ X ] Setup project repo with folder structure
- [ X ] Implement logging utility (TensorBoard / WandB / custom)
- [ X ] Implement deterministic seeding utility
- [ X ] Create base configuration YAML (`configs/base.yaml`)
- [ X ] Create experiment launcher (`experiments/run_experiment.py`)

---

## Phase 1 — Data Pipeline (1 week)

- [ X ] Implement tokenizer (character-level)
- [ X ] Implement dataset loader for:
    - WikiText-2
    - TinyStories
- [ ] Implement preprocessing pipeline (sequence batching, context windows)
- [ ] Verify input-output shapes for sanity check
- [ ] Create small sample dataset for overfitting tests

---

## Phase 2 — Core Model Implementation (2 weeks)

- [ ] Positional embeddings (vanilla)
- [ ] Multi-Head Self-Attention (manual implementation)
    - [ ] Q, K, V projections
    - [ ] Scaled dot-product attention
    - [ ] Causal mask
    - [ ] Head concatenation
- [ ] Feedforward network (MLP with GELU)
- [ ] Transformer block (Pre-LN)
- [ ] Decoder stack
- [ ] Weight tying (embedding ↔ output projection)
- [ ] Verify forward pass shapes
- [ ] Unit tests for block correctness

---

## Phase 3 — Training Loop (1 week)

- [ ] Implement optimizer (AdamW)
- [ ] Implement learning rate scheduler (cosine / linear warmup)
- [ ] Gradient clipping
- [ ] Checkpoint saving & resuming
- [ ] Train on tiny batch for overfitting sanity check
- [ ] Logging:
    - [ ] Train/Validation loss
    - [ ] Gradient norms
    - [ ] Parameter norms
- [ ] Verify reproducibility

---

## Phase 4 — Experiments (2 weeks)

### Baseline Experiments

- [ ] Depth vs Width study
- [ ] Learning rate schedule comparison
- [ ] Scaling study (0.5M → 5M parameters)
- [ ] Overfitting behavior for tiny batches

### Optional Advanced Experiments

- [ ] Attention head specialization
- [ ] Embedding geometry analysis (PCA / t-SNE)
- [ ] Gradient flow / norm visualization
- [ ] Compare Pre-LN vs Post-LN (later phase)

---

## Phase 5 — Analysis & Visualization (1 week)

- [ ] Attention map plotting
- [ ] Embedding manifold visualization
- [ ] Loss curves and gradient tracking plots
- [ ] Parameter scaling behavior plots
- [ ] Generate tables for report

---

## Phase 6 — Research Report / Documentation (1 week)

- [ ] Write technical report:
    - [ ] Introduction / Motivation
    - [ ] Architecture (mathematics + diagrams)
    - [ ] Dataset description
    - [ ] Training procedure
    - [ ] Experiments & results
    - [ ] Discussion & conclusions
- [ ] Prepare figures for report (`report/figures`)
- [ ] Update README with latest baseline / findings
- [ ] Prepare summary blog post / project showcase

---

## Phase 7 — Optional Future Extensions

- [ ] Add rotary positional embeddings
- [ ] Implement RMSNorm
- [ ] Swap GELU → SwiGLU in MLP
- [ ] Train LoRA / fine-tuning experiments
- [ ] Study efficiency: FlashAttention / sparse attention
- [ ] Multi-dataset scaling study
- [ ] Mechanistic interpretability on learned weights

---

