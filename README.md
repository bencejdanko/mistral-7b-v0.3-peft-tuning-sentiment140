# Mistral 7B v0.3 PEFT Tuning on Sentiment140

## Dataset: Sentiment140

[https://huggingface.co/datasets/bdanko/sentiment140](https://huggingface.co/datasets/bdanko/sentiment140)

* Binary Sentiment Classification (0: Negative, 4: Positive).
* Training set is 5,000 samples (shuffled and stratified).
* Test set is 1,000 samples (balanced $50/50$ distribution to ensure fair evaluation).
* Preprocessing is Mapping label `4` to `1` for standard binary cross-entropy compatibility. Removal of data leakage by ensuring no overlap between training and test indices.

## Model

Model is https://huggingface.co/mistralai/Mistral-7B-v0.3.

### VRAM & PEFT Efficiency

Full fine-tuning would require 112 GB of VRAM for a 7B model (weights + gradients + AdamW states). With Parameter-Efficient Fine-Tuning (PEFT) it's more feasbile. Mistral-7B has ~7.3 billion parameters. At bfloat16 precision (2 bytes/param), the base weights occupy ~14.6 GB.

By using LoRA or Adapters, we only train < 2% of the total parameters (roughly 50M - 150M params).

At BF, we require ~20-24 GB, an A10, or RTX 3090/4090 would work.

## Methods

### LoRA (Low-Rank Adaptation)

Injects trainable low-rank matrices into the Transformer layers (specifically the $W_q$ and $W_v$ projections).
* target `q_proj`, `v_proj`
* Update weights via $\Delta W = A \times B$, where $A$ and $B$ are low-rank.

### 2. Adapters
Injects small bottleneck layers after the Feed-Forward Network (FFN) or Attention layers.
* **Architecture:** Down-projection $\rightarrow$ Non-linearity $\rightarrow$ Up-projection.
* **Implementation:** Using `adapter-transformers` or `PEFT` library integration.

---

## Hyperparameter Tuning (Optuna)

We use Optuna to maximize the F1-Score. We will run 20 trials per method.

### Search Space & Justification

| Method | Parameter | Search Space | Justification |
| :--- | :--- | :--- | :--- |
| **LoRA** | Rank ($r$) | $\{4, 8, 16, 32\}$ | Higher $r$ captures more complexity but increases VRAM. |
| | Alpha ($\alpha$) | $\{16, 32, 64\}$ | Scaling factor for the learned weights. |
| | Learning Rate | $[1 \times 10^{-5}, 5 \times 10^{-4}]$ | Critical for convergence speed and stability. |
| **Adapters**| Bottleneck Dim | $\{32, 64, 128\}$ | Controls the capacity of the bottleneck layer. |
| | Learning Rate | $[5 \times 10^{-5}, 1 \times 10^{-3}]$ | Adapters often tolerate higher rates than LoRA. |
| | Dropout | $[0.0, 0.3]$ | Prevents overfitting on the small 5k dataset. |

**Compute Budget:** Total of 40 trials. Estimated 4-6 hours on an NVIDIA L4 (GCP/Colab).

---

## Evaluation Metrics
We evaluate the classification performance using the following:
* **Accuracy:** Overall correctness.
* **Precision:** Quality of positive predictions.
* **Recall:** Ability to find all positive instances.
* **F1-Score:** Harmonic mean of Precision and Recall (**Primary Metric**).

**Target Goal:** $> 93\%$ F1-Score (Bonus Tier).

## Results Table

| Method | Best Hyperparameter Set | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base LLM** | N/A (Zero-Shot) | | | | |
| **LoRA (FT)** | $r=X, \alpha=Y, lr=Z$ | | | | |
| **Adapters (FT)** | dim=$A$, lr=$B$, drop=$C$ | | | | |

## Deliverables

### Models
* `bdanko/gemma-2b-sentiment-lora`: Best LoRA adapter weights.
* `bdanko/gemma-2b-sentiment-adapters`: Best Adapter weights.

### Raw Data
* `bdanko/peft-sentiment-optuna-study`: Exported CSV of all 40 trials and their respective metrics.

## Qualitative Analysis
* **Adapter vs. LoRA:** Comparison of training stability and VRAM usage.
* **Error Analysis:** Review of 3 samples where the model misclassified sentiment (e.g., sarcasm or double negatives) and how the PEFT methods handled them differently.