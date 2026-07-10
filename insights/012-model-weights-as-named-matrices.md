---
number: 12
title: "Model Weights as Named Matrices"
book: "Inference Engineering"
tags: ["weights","matrices","checkpoint","embedding","attention","feedforward","unembedding"]
---

# Insight 12: Model Weights as Named Matrices

### Key Idea
A model's weights are individual learned numbers. Those numbers are organized into named matrices and tensors, and each matrix has a specific job in the transformer forward pass.

### Why It Works
When people say a model has billions of weights, they mean it has billions of learned scalar values. These values are not used as one giant flat list. They are grouped into structured matrices such as embedding, query, key, value, output, up-projection, down-projection, and unembedding matrices.

A useful mental model is:

```text
weight = one learned number
matrix/tensor = a named group of learned numbers
checkpoint = a structured object containing all learned matrices/tensors
```

Training fills in the numbers inside these matrices. Inference loads those matrices and uses them at the correct point in the forward pass.

### Example
```ts
type ModelWeights = {
  embedding: {
    tokenEmbeddingTable: Matrix;
  };

  layers: TransformerLayerWeights[];

  unembedding: {
    outputProjection: Matrix;
  };
};

type TransformerLayerWeights = {
  attention: {
    query: Matrix;
    key: Matrix;
    value: Matrix;
    output: Matrix;
  };

  feedForward: {
    upProjection: Matrix;
    downProjection: Matrix;
  };

  normalization: {
    scale: Vector;
    shift: Vector;
  };
};
```

This type is the useful high-level shape:

```text
ModelWeights
├── embedding
├── many transformer layers
│   ├── attention matrices
│   ├── feedforward matrices
│   └── normalization parameters
└── unembedding
```

### Matrix Roles

**Embedding matrix:** a learned lookup table that maps each vocabulary token ID to its initial embedding vector. During training, the model learns directions in embedding space so tokens with related meanings, roles, or usage patterns tend to point in similar directions.

**Query matrix:** turns a token's hidden vector into a query vector: what this token is looking for.

**Key matrix:** turns a token's hidden vector into a key vector: what this token advertises for matching.

**Value matrix:** turns a token's hidden vector into a value vector: what information this token provides if attended to.

**Attention output matrix:** combines the outputs of attention heads and projects them back into the main hidden-vector size.

**Up-projection matrix:** expands the hidden vector into a larger feedforward feature space.

**Down-projection matrix:** compresses the expanded feedforward vector back to the model's main hidden-vector size.

**Unembedding matrix:** maps the final hidden vector into one logit score for every token in the vocabulary.

### Parameter Count Intuition
For a GPT-3 175B-style model, the rough parameter counts are:

```text
Embedding matrix:          ~617M
Query matrices:            ~14B
Key matrices:              ~14B
Value matrices:            ~14B
Attention output matrices: ~14B
Up-projection matrices:    ~57B
Down-projection matrices:  ~57B
Unembedding matrix:        ~617M
```

The query, key, and value matrices are relatively small per attention head, but they add up because there are many heads and many layers.

The up-projection and down-projection matrices are especially large because the feedforward network expands the hidden vector into a much larger space and then compresses it back.

### When It Matters
This matters when separating learned parameters from inference-time activations.

Learned during training:

```text
embedding matrix
query matrix
key matrix
value matrix
attention output matrix
up-projection matrix
down-projection matrix
unembedding matrix
```

Created during inference:

```text
token embeddings
hidden vectors
query vectors
key vectors
value vectors
attention weights
logits
KV cache
```

The matrices are the learned machinery. The vectors are the temporary values produced when data flows through that machinery.

### Mental Model
> The checkpoint is a structured cabinet of learned matrices. Each matrix is a tool used at a specific point in the transformer.

### Takeaway
Model weights are not mysterious blobs. They are learned numbers organized into named matrices. Each matrix has a job: embeddings move tokens into model space, attention matrices help tokens exchange information, feedforward matrices transform hidden vectors, and the unembedding matrix turns the final hidden vector back into vocabulary logits.
