---
number: 7
title: "Attention in the Inference Flow"
book: "Inference Engineering"
tags: ["attention","kv-cache","prefill","decode","transformer","logits"]
---

# Insight 7: Attention in the Inference Flow

### Key Idea
Attention is an internal operation inside the transformer that helps produce context-aware hidden vectors. It is used during both prefill and decode, and it is one of the mechanisms that makes logits possible.

### Why It Works
During prefill, the prompt tokens are sent through the transformer. Inside each transformer layer, self-attention lets tokens relate to other tokens in the same sequence. The transformer produces two important results: logits for the next token and a KV cache containing reusable keys and values for the prompt tokens.

During decode, the newly generated token is sent back through the same transformer along with the existing KV cache. Each layer uses its own cached keys and values from previous tokens. The new token creates a temporary query, compares that query against cached keys, uses the resulting attention weights to mix cached values, and produces an updated hidden vector.

The attention weights themselves are temporary and are not what gets stored in the KV cache. The KV cache stores keys and values, organized per layer, per token, and usually per attention head. Queries are computed fresh for the current token and are not cached.

### Example
```ts
function prefill(promptTokens: Token[]) {
  // Process the whole prompt once.
  return transformerForward({
    tokens: promptTokens,
    kvCache: null,
    mode: "prefill",
  });
}

function decode(newToken: Token, kvCache: KVCache) {
  // Process only the newest token, while reusing prior K/V.
  return transformerForward({
    tokens: [newToken],
    kvCache,
    mode: "decode",
  });
}

function transformerForward(args: {
  tokens: Token[];
  kvCache: KVCache | null;
  mode: "prefill" | "decode";
}) {
  let vectors = embedTokens(args.tokens);
  let updatedKvCache = args.kvCache ?? emptyKvCache();

  for (const layer of transformerLayers) {
    const attentionResult = selfAttention({
      vectors,
      layerCache: updatedKvCache[layer.id],
      layer,
    });

    // Main output of attention: updated hidden vectors.
    vectors = attentionResult.updatedVectors;

    // Side output of attention: new K/V saved for future tokens.
    updatedKvCache[layer.id] = attentionResult.updatedLayerCache;

    vectors = feedForward(vectors, layer);
  }

  const lastVector = vectors[vectors.length - 1];
  const logits = projectToVocabulary(lastVector);

  return {
    logits,
    kvCache: updatedKvCache,
  };
}

function selfAttention(args: {
  vectors: Vector[];
  layerCache: LayerKVCache;
  layer: TransformerLayer;
}) {
  const queries = args.vectors.map(v => makeQuery(v, args.layer));
  const newKeys = args.vectors.map(v => makeKey(v, args.layer));
  const newValues = args.vectors.map(v => makeValue(v, args.layer));

  const allKeys = [...args.layerCache.keys, ...newKeys];
  const allValues = [...args.layerCache.values, ...newValues];

  const updatedVectors = queries.map(query => {
    const attentionWeights = softmax(
      allKeys.map(key => dotProduct(query, key))
    );

    return weightedSum(allValues, attentionWeights);
  });

  return {
    updatedVectors,
    updatedLayerCache: {
      keys: allKeys,
      values: allValues,
    },
  };
}
```

### When It Matters
This matters when understanding how prefill and decode connect. Prefill creates the initial KV cache for the prompt. Decode reuses that cache so the model does not need to recompute keys and values for all previous tokens every time it generates another token.

It also matters for understanding the boundary between the transformer and the inference engine. Attention happens inside the transformer before logits are produced. The inference engine receives logits afterward and applies sampling, temperature, top-k, top-p, and other generation controls.

### Mental Model
> Attention updates the current token's hidden vector by searching over cached keys and mixing the corresponding values. The KV cache is the memory attention reads from, not the attention result itself.

### Takeaway
Attention is used in both prefill and decode. Its main output is an updated hidden vector that continues through the transformer toward logits. Its side effect is updating the KV cache by saving the new token's keys and values for future decode steps.
