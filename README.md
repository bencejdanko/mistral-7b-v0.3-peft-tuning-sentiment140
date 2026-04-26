# Mistral-7B-v0.3 PEFT Tuning on Sentiment140

When fine-tuning the models, it can often be infeasible to continue full-weight training on all parameters, which may require 4-5 times the memory in order to store the optimizer states, full precision, and gradients. A frugal approach is to instead focus on training a smaller subset of parameters that can influence the model and can achieve the same results as a full fine tune.

We test LoRA and Bottleneck Adapter techniques for model tuning. LoRA uses low-rank decomposition to approximate weight updates. Bottleneck adapters freeze the entire pre-trained backbone and only train a set of small, newly introduced modules, inserting small feed-forward network inserted into each layer of the Transformer.

Instead of a generative output, this a causal transformer binary classifier. We define the model configuration below with `transformers`:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,                           # mistralai/Mistral-7B-v0.3
    num_labels=2,                       # 2 output labels, binary classification. However, using 1 and making this a regression problem is feasible.
    dtype=torch.bfloat16,               # tensor type configuration - we use the original BF16
)
```

## Final Results

| Method | Best Hyperparameters Set | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base LLM | | 0.5380 | 0.5271 | 0.7380 | 0.6150 |
| LoRA (Fine-tuned) | {'r': 32, 'alpha': 32, 'lr': 0.0004959615908352659} | **0.8780** | **0.8826** | **0.8720** | **0.8773** |
| Adapters (Fine-tuned) | {'reduction_factor': 16, 'dropout': 0.17640177247960154, 'lr': 0.00025773023366483446} | 0.6390 | 0.6387 | 0.6400 | 0.6393 |

**We trained for 5 epochs on the best optuna-study params*

[Full online report (Google Documents)](https://docs.google.com/document/d/1gmUemWx8zt6N7PIbGn-L2yHQA1rUb76D09YVTAsJqsE/edit?usp=sharing).

