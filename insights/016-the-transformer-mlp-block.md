---
number: 16
title: "The Transformer MLP Block"
book: "Inference Engineering"
tags: ["mlp","feedforward","ffn","backpropagation","hidden-vectors","transformer"]
---

# Insight 16: The Transformer MLP Block

### Key Idea
The MLP, also called the feedforward network or FFN, is the familiar neural-network graph with input nodes, hidden neurons, and output nodes. Inside a transformer layer, it transforms each token's hidden vector independently using the same learned weights.

### Why It Works
For one token, the number of input nodes equals the hidden-vector dimension, usually called `d_model`.

The MLP commonly expands that vector into a larger intermediate layer, applies an activation function, and then projects it back down to the original hidden dimension.

```text
input:        d_model dimensions
hidden layer: larger intermediate dimension
output:       d_model dimensions
```

For example:

```text
12,288 input values
→ 49,152 intermediate neurons
→ 12,288 output values
```

The output has the same dimension as the input so it can continue through the transformer stack and be combined with the existing hidden-vector stream.

The MLP is an important place where the model stores and applies learned features and knowledge. It is not literally a fact database, but its learned weights help detect and transform useful patterns inside a token's context-aware hidden vector.

During prefill, every prompt token goes through the same MLP weights independently. During decode, the newest token goes through those same MLP weights after attention.

### Example
```ts
function runTransformerMlpForOneToken(
  hiddenVector: Vector,
  weights: MlpWeights
): Vector {
  // Input size: d_model, such as 12,288 dimensions.
  const expandedVector = multiply(
    hiddenVector,
    weights.upProjection
  );

  // Intermediate size is often much larger, such as 49,152 dimensions.
  const activatedFeatures = activation(expandedVector);

  // Project back to d_model so the vector can continue through the transformer.
  const mlpOutput = multiply(
    activatedFeatures,
    weights.downProjection
  );

  return add(hiddenVector, mlpOutput);
}
```

Conceptually, each token goes through its own instance of this computation:

```text
"fluffy" hidden vector  → same MLP graph → updated "fluffy" vector
"blue" hidden vector    → same MLP graph → updated "blue" vector
"creature" hidden vector → same MLP graph → updated "creature" vector
```

The activations differ for each token because their hidden vectors differ, but the learned MLP weights are shared.

### When It Matters
This matters when separating attention from the MLP.

Attention lets tokens exchange information with other tokens. The MLP then transforms the features inside each token's resulting hidden vector independently.

It also matters when separating training from inference. Backpropagation happens during training, where gradients update the MLP's weights. During normal inference, the weights are fixed and the model only performs the forward computation.

### Mental Model
> Attention lets tokens communicate. The MLP is the familiar neural-network graph that processes each token's context-aware vector on its own.

### Takeaway
The transformer MLP takes a `d_model`-dimensional hidden vector, expands it through a larger hidden layer, and returns another `d_model`-dimensional vector. Its weights are learned through backpropagation during training and reused unchanged during inference.
