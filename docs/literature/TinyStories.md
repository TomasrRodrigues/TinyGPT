# TinyStories — Eldan & Li (2023)

1. [Motivation](#motivation)
2. [Key Hypothesis](#key-hypothesis)
3. [Dataset Construction](#dataset-construction)
4. [Dataset Statistics](#dataset-statistics)
5. [Training Setup in Paper](#training-setup-in-paper)
6. [Evaluation Method](#evaluation-method)
7. [Why This Matters for TinyGPT](#why-this-matters-for-tinygpt)
8. [Implementation-Relevant Notes](#implementation-relevant-notes)


## Motivation

**Modern language models** achieve strong performance mainly through scale, this is larger architectures trained on massive and diverse corpora. This work challenges this paradigm by examining whether it is possible to produce a coherent, grammatically correct and semantically meaningful text with a small model.

Existing benchmarks are often derived from large, heterogeneous corpora such as Wikipedia or web-scale crawls. These datasets contain broad vocabulary, complex structures and long-range dependencies. While suitable for evaluating large models, we don't know if small models fail because they are too small or because the data distribution is too complex.

TinyStories works differently. Instead of filtering natural data, the authors construct a synthetic dataset designed to retain core elements of natural language while dramatically constraining vocabulary and conceptual breadth to that typically understood by 3-4-year old children.

This controlled simplification allows the authors to investigate whether:
- small transformer models can internalize syntax and semantic coherence when distributional complexity is reduced
- reasoning behaviors can emerge without massive scale
- dataset design, rather than parameter count alone, is a critical factor in language modeling performance.

The work challenges the implicit assumptions that scaling is the only viable path toward coherent language generation and proposes a research direction: **dataset refinement as a tool for probing model capability**


## Key Hypothesis

As stated in the motivation, the key hypothesis brought with this paper is that it is possible to produce coherent and fluent text, following the important rules of language (syntatic and semantic coherence, logical consistency, grammar), without requiring hundreds of millions of parameters.

The authors hypothesize that the poor performance of small models is largely attributable to the breadth and heterogeneity of standard corpora rather than to an inherent inability to learn language structure. 

If a dataset preserves the essential elements of natural language but restricts the vocabulary and factual knowledge base to that of a 3-4-year old child, models with fewer than 10 million parameters can learn to generate fluent, consistent stories and demonstrate reasoning.

This suggests that model capability is strongly dependent on the match between model capacity and dataset complexity, while remaining substantially more parameter-efficient.


## Dataset Construction

The authors utilize OpenAI's GPT-3.5 and GPT-4 models to generate the synthetic dataset. Story generation is guided by a controlled vocabulary of approximately 1500 basic words intended to reflext the linguistic knowledge og a typical 3-4-year old child. The vocabulary is organized into lexical categories (e.g., nouns, verbs, adjectives) to facilitate structured prompting.

To encourage lexical diversity, compositional generation and avoid repetitive outputs, each generation prompt randomly selected three words (one noun, one verb, one adjective) that must be combined into the story. This mechanism reduces repetitive outputs and prevents the model from collapsing into highly frequent narrative patterns.

Additionally, prompts require a randomly selected subset of narrative features, such as dialogue, a plot twist, a bad ending or a moral value. These constraints introduce structureal variation beyond vocabulary-level diversity.

The authors also introduce *TinyStories-Instruct*, an instruction-following variant in which stories are preceded by randomly generated instruction combinations (e.g., required words, mandatory sentences, specified features, or short summaries). This variant enables evaluation of instruction-following capabilities in small models.


## Dataset Statistics

The dataset vocabulary is designed to approximate the comprehension level of a typical 3–4-year-old child, consisting of roughly 1,500 core lexical items.

Each instance in the dataset is a short story composed of 2–3 paragraphs, typically featuring simple sentence structures and limited factual complexity.

- Total stories: $\approx 2.2$ million
- Train/validation split: $80%$ train / $20%$ validation
- Total tokens $\approx 100M - 500M$ tokens
- Average story length: $\approx 100 - 400$ tokens
- Maximum story length: $512$ tokens (based on the models' maximum context window)
- File format: `.parquet` (via Hugging Face)
- Paragraphs separated by newline characters
- Character set: English (UTF-8)


## Training Setup in Paper

The authors trained a variety of small language models (SLMs) to test the efficacy of the TinyStories dataset, keeping the architecture and training pipeline extremely resource-efficient:
- **Base Architecture**: The models utilize the GPT-Neo architecture.
- **Context and Window Size**: The models were configured with a context length of 512 and a window size of 256.
- **Tokenizer Customization**: They used the standard GPT-Neo tokenizer but heavily restricted the vocabulary by keeping only the top 10,000 most common tokens.
- **Model Scale**: The parameter counts of the tested models ranged broadly from roughly 1 million to 80 million parameters. Model depth was also varied significantly, ranging from ultra-shallow 1-layer transformer blocks up to 12 layers.
- **Hardware & Compute**: Emphasizing the accessibility of this approach, training these generative models typically took less than a day  (at most 30 hours) using a single V100 GPU.

## Evaluation Method

Standard NLP benchmarks (e.g., LAMBDA, TriviaQA, or Winograd Schema Challenge) typically require a model to output a single word or a highly structured short phrase. Beacause this approach fails to capture the richness, fluency and diversity required for natural language generation, the authors introduced a novel evaluation paradigm: **GPT-Eval**.

This framework utilizes a frontier model (GPT-4) to act as a "human teacher" grading a student's creative writing assignment. The pipeline operates as follows:
- **Prompting**: The SLM is provided with the beginning of a story drawn from a manually constructed evaluation set of about 50 prompts. These prompts typically cut off a sentence in the middle to test grammatical continuation.
- **Generation**: For each prompt, the SLM generates 10 different completions using a generation of temperature 1
- **Multidimensional Grading**: GPT-4 is given the original prompt and the SLM's completion. It is instructed to evaluate the text and provide quantitative scores across several dimensions: **grammar**, **creativity** and **consistency** with the beginning of the story
- **Instruction-Following Evaluation**: For models trained on the *TinyStories-Instruct* variant, GPT-4 is also given the prompt instructions and evaluates the generated story based on how accurately it reflects the requested constraints (consistency) and the overall coherence of the plot
- **Final Scoring**: The final score for a model on a given metric is the average of the GPT-4 evaluation scores across all completions.


## Why this Matters for TinyGPT

It provides empirical proof that parameter count is not the sole bottleneck for fluency; highly constrained models can generate coherent text if the training distribution is tightly controlled.

It highlights a critical architectural trade-off where model depth (number of layers) is more important for context-tracking and content consistency.

Conversely, it shows that model width (embedding dimension) is more crucial for capturing factual knowledge.

It demonstrates that fundamental scaling laws regarding optimal model size and training compute (FLOPs) still apply at the micro-scale level.

It reveals that smaller, constrained networks exhibit highly interpretable features, such as distinct semantic and local attention heads, which can simplify debugging and analysis.

## Implementation-Relevant Notes

When computational budgets are heavily constrained, increasing the number of attention heads consistently improves performance across all metrics. Grammar and stbtax are mastered at relatively small models sizes and plateau early in training.

Consistency and creativity require larger sizes and deeper networks to fully emerge. A single transformer layer struggles substantially with instruction-following tasks.

Upgrading to just two layers is a sufficient to achieve a basic level of instruction adherence.
