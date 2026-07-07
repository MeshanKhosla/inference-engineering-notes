---
number: 5
title: "Decoding Strategies: Temperature, Top-k, Top-p, and Sampling"
book: "Inference Engineering"
tags: ["decoding","temperature","top-k","top-p","sampling","logits"]
---

# Insight 5: Decoding Strategies: Temperature, Top-k, Top-p, and Sampling

### Key Idea
The transformer does not directly choose the next token. It produces logits, which are raw scores for every token in the vocabulary. The inference engine then turns those logits into a probability distribution and samples the next token.

### Why It Works
After the transformer outputs logits, the inference engine can apply different decoding strategies before choosing a token.

Temperature modifies the logits before softmax. Lower temperature exaggerates the differences between logits, making the model more deterministic. Higher temperature shrinks the differences, making the distribution flatter and allowing less likely tokens to be sampled more often.

Top-k keeps only the top `k` most likely tokens and removes the rest before sampling.

Top-p, also called nucleus sampling, keeps the smallest set of tokens whose cumulative probability reaches `p`. This adapts to the model's confidence: if the model is very confident, only a few tokens are considered; if the model is uncertain, more tokens are allowed.

### When It Matters
These strategies matter whenever you want to control the style of generation.

Low temperature is useful for factual, deterministic, or structured tasks.

Higher temperature is useful for brainstorming, creative writing, and tasks where variety is valuable.

Top-k and top-p are useful for preventing the model from sampling extremely unlikely or low-quality tokens while still allowing some diversity.

### Mental Model
> The transformer produces a weighted menu of possible next tokens. Decoding strategies decide how adventurous the sampler is allowed to be when choosing from that menu.

### Example
```javascript
function forwardPass(tokens, kvCache = null, temperature = 1.0) {
  const result = transformer(tokens, kvCache);

  const adjustedLogits = result.logits.map(
    logit => logit / temperature
  );

  const probabilities = softmax(adjustedLogits);

  const filteredProbabilities = applyTopKOrTopP(probabilities);

  const nextToken = sample(filteredProbabilities);

  return {
    nextToken,
    kvCache: result.kvCache,
  };
}
```

### Takeaway
If you always picked the highest-probability token, temperature, top-k, and top-p would not matter. These controls matter because generation usually samples from a probability distribution rather than always choosing the top token.
