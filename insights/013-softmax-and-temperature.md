---
number: 13
title: "Softmax and Temperature"
book: "Inference Engineering"
tags: ["softmax","temperature","logits","sampling","probabilities","decoding"]
---

# Insight 13: Softmax and Temperature

### Key Idea
Softmax turns raw scores into a normalized probability distribution. Temperature modifies softmax by dividing each score by `T` before exponentiating, which controls how rigid or random the distribution becomes.

### Why It Works
Raw logits or attention scores are just unnormalized numbers. They can be positive, negative, large, or small. Softmax converts those scores into relative weights where every output value is between `0` and `1`, and all output values sum to `1`.

Formula without temperature:

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j = 0 to n - 1
```

Meaning:

```text
1. Take e raised to each score.
2. Add up all those exponentiated scores.
3. Divide each exponentiated score by the total.
```

The exponential makes larger scores grow faster than smaller scores. So softmax preserves the ranking, but exaggerates differences.

For example:

```text
scores = [2, 1, 0]
softmax(scores) ≈ [0.665, 0.245, 0.090]
```

The highest score gets the most probability, but the lower scores are not completely erased. This is why it is a soft winner-picking function.

### Temperature
Temperature is added by dividing each score by `T` inside the exponent:

```text
softmax_with_temperature(x_i) = exp(x_i / T) / sum(exp(x_j / T)) for j = 0 to n - 1
```

Lower temperature makes the distribution sharper and more rigid:

```text
T < 1
```

Dividing by a small number makes score differences larger. The highest-scoring token becomes much more likely, and lower-scoring tokens become much less likely. As temperature approaches `0`, the behavior approaches always picking the highest-scoring token.

Higher temperature makes the distribution flatter and more random:

```text
T > 1
```

Dividing by a larger number shrinks score differences. Lower-scoring tokens get more probability than they otherwise would, so sampling becomes more varied.

### When It Matters
Softmax appears in two important places:

- In attention, it turns query/key dot-product scores into attention weights.
- In generation, it turns logits into probabilities over vocabulary tokens.

Temperature matters during generation because it changes how adventurous the sampler is allowed to be without changing the model's trained weights.

### Mental Model
> Softmax turns a scoreboard into a probability budget. Temperature controls how rigid or loose that budget is.

Cold temperature is rigid: it strongly favors the highest score.

Hot temperature is loose: it allows more randomness.

### Takeaway
Softmax normalizes raw scores into probabilities. Temperature is grounded directly in the softmax math: each score is divided by `T` before exponentiation. Lower temperature sharpens the distribution and makes generation more deterministic; higher temperature flattens the distribution and makes generation more random.
