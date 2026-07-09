---
number: 10
title: "Query Vectors and Attention"
book: "Inference Engineering"
tags: ["query-vector","attention","wq","transformer-layer","attention-heads","training-vs-inference"]
---

# Insight 10: Query Vectors and Attention

### Key Idea
A query vector is a temporary inference-time vector that represents what information a token is looking for. It is produced by multiplying the token's current hidden vector by a learned query weight matrix, usually written as `W_Q`.

### Why It Works
A token first starts as a learned token embedding. Then position information is added, producing an initial hidden vector for that token at that position.

For example, in the sentence:

```text
a fluffy blue creature roamed the verdant forest
```

the token `"creature"` starts with its learned embedding. After position information is added, it becomes the initial hidden vector for `"creature"` at that location in the sentence.

Inside a transformer layer, that hidden vector is multiplied by a learned query matrix:

```ts
const tokenEmbedding = embeddingTable["creature"];
const positionVector = positionEmbedding(positionOf("creature"));

const hiddenVector = add(tokenEmbedding, positionVector);

const queryVector = multiply(hiddenVector, W_Q);
```

`W_Q` is learned during training. After training, it is fixed. During inference, the model uses `W_Q` to transform the current hidden vector into a query vector.

Conceptually, the query vector acts like a question the token is asking. For `"creature"`, one attention head might learn a query like:

```text
Which earlier words describe me?
```

That query could match strongly with keys for:

```text
fluffy
blue
```

This does not mean the model literally stores an English question. The query is a vector, not text. But the vector can behave like a learned search request.

### Example
```ts
type Vector = number[];
type Matrix = number[][];

function createInitialHiddenVector(token: Token, position: number): Vector {
  const tokenEmbedding = embeddingTable[token];
  const positionVector = positionEmbedding(position);

  return add(tokenEmbedding, positionVector);
}

function createQueryVector(hiddenVector: Vector, attentionHead: AttentionHead): Vector {
  // W_Q is learned during training.
  // The query vector is computed during inference.
  return multiply(hiddenVector, attentionHead.W_Q);
}

const sentence = [
  "a",
  "fluffy",
  "blue",
  "creature",
  "roamed",
  "the",
  "verdant",
  "forest",
];

const creatureHiddenVector = createInitialHiddenVector(
  "creature",
  sentence.indexOf("creature")
);

const creatureQuery = createQueryVector(
  creatureHiddenVector,
  layer0.attentionHeads[0]
);

// Conceptually, this head's query might behave like:
// "Which earlier words describe this noun?"
```

### Where It Lives
A transformer layer usually contains a multi-head self-attention module. Each attention head can have its own learned query, key, and value matrices:

```ts
type AttentionHead = {
  W_Q: Matrix;
  W_K: Matrix;
  W_V: Matrix;
};

type TransformerLayer = {
  attentionHeads: AttentionHead[];
  feedForward: FeedForwardNetwork;
};
```

So `W_Q` is not one universal matrix for the whole model. Conceptually, there is usually a different `W_Q` per transformer layer and per attention head.

### When It Matters
This matters when separating learned parameters from temporary inference-time vectors.

Training learns:

```text
W_Q
W_K
W_V
embedding tables
feedforward weights
output projection weights
```

Inference computes:

```text
hidden vectors
query vectors
key vectors
value vectors
attention weights
logits
```

The query vector is not itself a trained weight. It is created during inference by applying a trained weight matrix to the current hidden vector.

### Mental Model
> `W_Q` is the learned machine. The query vector is the search request that machine produces for a specific token in a specific context.

### Takeaway
A query vector is how a token asks what information it needs from other tokens. `W_Q` is learned during training, but the query vector is computed during inference from the token's current hidden vector. Different attention heads can learn different kinds of questions, such as looking for adjectives, subjects, references, or other useful relationships.
