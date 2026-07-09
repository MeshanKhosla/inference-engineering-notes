---
number: 8
title: "Token Embeddings and Position Information"
book: "Inference Engineering"
tags: ["embeddings","tokens","position-embeddings","transformer","hidden-vectors"]
---

# Insight 8: Token Embeddings and Position Information

### Key Idea
A token's initial embedding is a learned lookup-table vector. The same vocabulary token starts with the same embedding every time, but the transformer also adds position information so the model knows where that token appears in the sequence.

### Why It Works
The tokenizer converts text into token IDs. The model then uses those token IDs to look up learned embedding vectors. For example, a token like `"fluffy"` has a fixed learned starting vector after training.

That initial embedding by itself only represents the token's general learned meaning. To make order matter, the model also adds or otherwise incorporates position information. This creates the initial hidden vector that enters the transformer stack.

After this, the vector flows through repeated transformer layers. Later operations, including attention and feedforward layers, modify the vector based on context. Those details can be studied separately; the important idea here is that the starting point is a lookup-table embedding plus position information.

### Example
```ts
function transformerForward(tokens: Token[]) {
  let vectors = tokens.map((token, position) => {
    const tokenVector = embeddingTable[token];
    const positionVector = positionEmbedding(position);

    return add(tokenVector, positionVector);
  });

  // Small models may have fewer than 20 transformer layers; big models can have over 100.
  for (const layer of transformerLayers) {
    vectors = selfAttention(vectors, layer);
    vectors = feedForward(vectors, layer);
  }

  return vectors;
}
```

### When It Matters
This matters when separating the beginning of the model from the rest of the transformer. The tokenizer does not produce meanings directly; it produces token IDs. The embedding table turns those IDs into vectors. Position information then tells the model where each token appears.

It also matters when understanding why the same token can behave differently in different sentences. The initial embedding for `"fluffy"` is the same every time, but once position and later transformer layers modify it, the hidden vector becomes context-dependent.

### Mental Model
> The embedding table gives each token its starting meaning. Position information tells the model where that token is standing in the sentence. The transformer layers then reshape that starting vector using context.

### Takeaway
A token starts as a fixed learned embedding lookup, then position information is baked in. After that, the vector moves through the transformer stack, where later mechanisms such as attention modify it into a context-aware hidden vector.
