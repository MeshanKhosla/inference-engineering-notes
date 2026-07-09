---
number: 9
title: "Hidden Vectors"
book: "Inference Engineering"
tags: ["hidden-vectors","embeddings","vocabulary","context","logits","transformer"]
---

# Insight 9: Hidden Vectors

### Key Idea
A hidden vector is a context-specific internal representation inside the transformer. Unlike vocabulary tokens, hidden vectors are continuous points in vector space and do not need to correspond to one exact token in the vocabulary.

### Why It Works
The vocabulary is a fixed set of discrete tokens that the tokenizer can produce and the model can output. For example, a tokenizer may contain separate tokens like `"fluffy"`, `"blue"`, and `"creature"`, but not a single token for `"fluffy blue creature"`.

That does not stop the transformer from representing the combined idea internally. The token `"creature"` starts as a learned embedding lookup, but after position information and transformer layers modify it, its hidden vector can carry context from nearby tokens like `"fluffy"` and `"blue"`.

For the sentence:

```text
A fluffy blue creature roamed the verdant forest
```

The model can build hidden vectors conceptually like:

```text
hiddenVector("creature") ≈ creature + fluffy + blue
hiddenVector("forest") ≈ forest + verdant
```

These are not new vocabulary tokens. They are temporary internal representations that encode blended meaning in high-dimensional space.

At the end of the transformer, the final hidden vector is projected back into vocabulary space to produce logits. Only then does the model return to scoring fixed vocabulary tokens.

### Example
```ts
const creatureEmbedding = embeddingTable["creature"];

// The embedding is the generic learned starting point for the token.
let creatureHiddenVector = add(
  creatureEmbedding,
  positionEmbedding(positionOf("creature"))
);

// Transformer layers modify the vector using context.
creatureHiddenVector = addContext(creatureHiddenVector, [
  hiddenVectorFor("fluffy"),
  hiddenVectorFor("blue"),
]);

// This does not create a new vocabulary token called "fluffy blue creature".
// It creates a context-specific internal vector.
const contextualMeaning = creatureHiddenVector;

// Later, the model projects a final hidden vector into vocabulary-sized logits.
const logits = projectToVocabulary(finalHiddenVector);
```

### When It Matters
This matters when separating vocabulary tokens from internal model representations. The model can only read and write tokens from its fixed vocabulary, but inside the transformer it can represent richer combinations of ideas as hidden vectors.

It also matters for understanding attention. Attention does not create new output tokens directly. It helps update hidden vectors so they carry context from relevant tokens.

### Mental Model
> Tokens are the model's input/output symbols. Hidden vectors are the model's internal meaning states.

### Takeaway
The vocabulary limits what the model can input and output, but it does not limit what the transformer can represent internally. A phrase like `"fluffy blue creature"` can exist as a hidden vector even if it is not a single vocabulary token.
