---
number: 6
title: "The Inference Engine Controls Generation"
book: "Inference Engineering"
tags: ["inference-engine","logits","logit-biasing","temperature","sampling","structured-outputs"]
---

# Insight 6: The Inference Engine Controls Generation

### Key Idea
The transformer's job ends when it produces logits: raw scores for every vocabulary token. The inference engine then decides how those logits are converted into the next generated token.

### Why It Works
Because logits are just numbers, they can be modified before they are converted into probabilities. This allows the inference engine to influence generation without changing the transformer or its learned weights.

Common inference-time operations include:

- Temperature: scales logits to make the probability distribution more or less confident.
- Logit biasing: manually increases, decreases, or forbids specific token scores.
- Top-k and top-p: filter which tokens are eligible for sampling.
- Sampling: selects the next token from the final probability distribution.

This is useful for structured outputs, schemas, and tool calling. For example, if the system expects JSON, the inference engine can bias the model toward valid JSON tokens or forbid invalid tokens.

### When It Matters
This matters whenever the application needs control over the model's output without retraining the model.

Examples include:

- Forcing valid JSON or schema-conforming output.
- Steering the model toward a tool call.
- Preventing specific tokens from being generated.
- Making outputs more deterministic or more creative.
- Applying application-level constraints at inference time.

### Mental Model
```text
Prompt
    ↓
Tokenizer
    ↓
Transformer
    ↓
Logits
    ↓
Inference Engine
    • Temperature
    • Logit Biasing
    • Top-k / Top-p
    • Sampling
    ↓
Next Token
```

### Takeaway
The transformer predicts what is likely. The inference engine decides how those predictions are used. Temperature, logit biasing, top-k, top-p, and sampling are all ways to control generation at inference time without changing the model itself.
