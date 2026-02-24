# TinyGPT

**TinyGPT** is a research-oriented, from-scratch PyTorch implementation of a decoder-only transformer. This project is designed to study **training dynamics, scaling behavior, and mechanistic properties**, while emphasizing modular engineering and reproducibility.

TinyGPT is optimized for **small-scale models (0.5M–5M parameters)** to allow rapid experimentation with architecture, optimization, and interpretability.


## Project Goals

1. Implement a fully modular Transformer from scratch with custom implementations of:
    - Multi-Head Self-Attention with causal masking
    - MLP blocks with activation functions (GELU/SwiGLU)
    - Positional embeddings (absolute or rotary)
    - Pre-LayerNorm architecture
    - Weight tying for embeddings and output projection
2. Conduct structured experiments to study:
    - Depth vs width trade-offs
    - Learning rate schedules and warmup strategies
    - Scaling behavior of small models
    - Gradient norms and stability
    - Attention patterns and embedding geometry
3. Reproducibility & Research Rigor
    - Fixed seeds and deterministic training
    - Config-driven experiments via YAML files
    - Versioned checkpoints and logging
    - Automated analysis of gradients, embeddings, and attention maps
4. Research Output
    - Produce figures, plots, and reports suitable for a research paper
    - Enable mechanistic interpretability studies on small models
5. Produce reproducible results and research-style reports.


## Repo Structure

```
tinygpt/
   ├── configs/         # Hyperparameter and experiment configuration files (YAML)
   ├── data/            # Dataset loading, tokenization, preprocessing scripts
   ├── model/           # Core model components: embeddings, attention, MLP, blocks, full transformer
   ├── training/        # Training loop, optimizers, schedulers, checkpointing
   ├── analysis/        # Scripts for attention visualization, embedding geometry, gradient tracking
   ├── utils/           # Logging, seeding, metrics, helper functions
   ├── experiments/     # Entry points for running experiments reproducibly
   ├── report/          # Generated figures and research reports
   ├── main.py          # CLI entry point for training or evaluation
   ├── requirements.txt # Python dependencies
   └── README.md        # Project overview and documentation
```

**Design Philosophy**:
- Clear separation of concerns: model, training, analysis, experiments
- Modular and extensible components for experimentation
- Full reproducibility via configs, seeds, and logging


## Installation

```bash
git clone https://github.com/TomasrRodrigues/TinyGPT.git
cd TinyGPT
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Recommended: use *Python 3.10+* and *PyTorch >= 2.1* for optimal performance.
 

## License

MIT License - free for research and educational purposes.

Attribution appreciated for any derivative work or publication.
