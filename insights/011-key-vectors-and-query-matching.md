---
number: 11
title: "Key Vectors and Query Matching"
book: "Inference Engineering"
tags: ["key-vector","query-vector","attention","dot-product","wq","wk","training-vs-inference"]
---

# Insight 11: Key Vectors and Query Matching

### Key Idea
A key vector is a temporary inference-time vector that represents what a token offers for matching. A query asks what information is needed, and a key advertises whether a token can answer that query.

### Why It Works
Inside an attention head, the token's hidden vector is projected into a query vector using `W_Q`, and each token's hidden vector is projected into a key vector using `W_K`.

`W_Q` and `W_K` are learned during training. They are trained together so useful query/key relationships tend to align in the same attention-head vector space.

For example, in the sentence:

```text
a fluffy blue creature roamed the verdant forest
```

one attention head may learn a query for `"creature"` that behaves like:

```text
Which earlier words describe me?
```

The keys for `"fluffy"` and `"blue"` may behave like:

```text
I am a descriptive adjective before a noun.
```

The model then compares the query for `"creature"` against the keys for the tokens it is allowed to attend to. A dot product measures how aligned the vectors are. If the query and key point in similar directions, the dot product is high. If they do not align, the dot product is lower.

The key is not literally an English answer. It is a vector that can match a query vector. But the mental model is useful: the query asks a question, and the key advertises whether that token can answer it.

### Example
```ts
function computeAttentionScoresForCreatureDuringPrefill(
  hiddenVectors: Record<Token, Vector>,
  attentionHead: AttentionHead
): Record<Token, number> {
  // During prefill, every token in the input produces a query.
  // Here we focus on the query for "creature".
  const creatureQuery = multiply(
    hiddenVectors["creature"],
    attentionHead.W_Q
  );

  // During prefill, every token in the input also produces a key.
  const keysByToken = {
    "a": multiply(hiddenVectors["a"], attentionHead.W_K),
    "fluffy": multiply(hiddenVectors["fluffy"], attentionHead.W_K),
    "blue": multiply(hiddenVectors["blue"], attentionHead.W_K),
    "creature": multiply(hiddenVectors["creature"], attentionHead.W_K),
    "roamed": multiply(hiddenVectors["roamed"], attentionHead.W_K),
    "the": multiply(hiddenVectors["the"], attentionHead.W_K),
    "verdant": multiply(hiddenVectors["verdant"], attentionHead.W_K),
    "forest": multiply(hiddenVectors["forest"], attentionHead.W_K),
  };

  // The dot product measures how aligned the query is with each key.
  // These scores are computed against every token the current token
  // is allowed to attend to.
  return {
    "a": dotProduct(creatureQuery, keysByToken["a"]),
    "fluffy": dotProduct(creatureQuery, keysByToken["fluffy"]),
    "blue": dotProduct(creatureQuery, keysByToken["blue"]),
    "creature": dotProduct(creatureQuery, keysByToken["creature"]),
    "roamed": dotProduct(creatureQuery, keysByToken["roamed"]),
    "the": dotProduct(creatureQuery, keysByToken["the"]),
    "verdant": dotProduct(creatureQuery, keysByToken["verdant"]),
    "forest": dotProduct(creatureQuery, keysByToken["forest"]),
  };
}

function convertAttentionScoresToWeights(
  scores: Record<Token, number>
): Record<Token, number> {
  // Softmax turns raw scores into weights that sum to 1.
  // Bigger scores become bigger weights.
  // This lets the model say "pay 60% attention here,
  // 25% there, 5% somewhere else," etc.
  return softmax(scores);
}

function computeAttentionWeightsForCreatureDuringPrefill(
  hiddenVectors: Record<Token, Vector>,
  attentionHead: AttentionHead,
  tokensInOrder: Token[]
): Record<Token, number> {
  const rawScores = computeAttentionScoresForCreatureDuringPrefill(
    hiddenVectors,
    attentionHead
  );

  const maskedScores = applyCausalMaskForDecoderOnlyModel(
    rawScores,
    "creature",
    tokensInOrder
  );

  return convertAttentionScoresToWeights(maskedScores);
}
```

### When It Matters
This matters when understanding how attention decides which tokens are relevant. The model does not manually know that `"fluffy"` and `"blue"` describe `"creature"`. Training shapes `W_Q` and `W_K` so that useful matches create high dot products during inference.

It also matters for understanding the difference between trained weights and temporary vectors. `W_Q` and `W_K` are learned during training. Query and key vectors are computed during inference from the current hidden vectors.

### Mental Model
> The query asks the question. The key advertises whether a token can answer it. The dot product checks whether they point in the same direction.

### Takeaway
Key vectors live in the same attention-head space as query vectors. Training learns `W_Q` and `W_K` together so related queries and keys tend to align. During inference, the dot product between a query and each key produces attention scores, which are then turned into attention weights.
