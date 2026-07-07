---
number: 4
title: "The Inference Loop: Prefill to Decode"
book: "Inference Engineering"
tags: ["inference","prefill","decode","kv-cache","autoregressive","transformer"]
---

# Insight 4: The Inference Loop: Prefill to Decode

### Key Idea
Inference happens in two main phases: prefill and decode. Prefill processes the entire prompt once. Decode generates the response one token at a time using the previously computed KV cache.

### Why It Works
When the prompt is first received, the model tokenizes the entire sequence and performs a full forward pass through the transformer. This produces logits for the next token and builds the initial KV cache.

After the first token is generated, the model does not need to reread the entire prompt from scratch. Instead, it feeds the newly generated token back into the transformer along with the KV cache. The KV cache represents reusable computation from all previous tokens.

This loop repeats until the model reaches a stopping condition, such as an end-of-sequence token or a maximum token limit.

### Example
Prompt:

`"the cat is"`

Tokenized as:

`["the", "cat", "is"]`

During prefill, the entire prompt is processed at once:

`forwardPass(["the", "cat", "is"])`

Suppose the model generates:

`"sleeping"`

During decode, the next call processes only the new token plus the KV cache:

`forwardPass(["sleeping"], kvCache)`

Then if it generates:

`"on"`

the next call is:

`forwardPass(["on"], updatedKvCache)`

### Mental Model
```javascript
function forwardPass(tokens, kvCache = null) {
  const result = transformer(tokens, kvCache);

  const logits = result.logits;
  const updatedKvCache = result.kvCache;

  const probabilities = softmax(logits);
  const nextToken = sample(probabilities);

  return {
    nextToken,
    kvCache: updatedKvCache,
  };
}

// ---------- Prefill ----------

const prompt = ["the", "cat", "is"];

let state = forwardPass(prompt);

// ---------- Decode ----------

while (state.nextToken !== "<eos>") {
  state = forwardPass([state.nextToken], state.kvCache);
}
```

### Takeaway
Prefill reads the entire prompt once. Decode extends the response one token at a time. The KV cache lets the model reuse previous computation, so it does not need to reread the entire prompt for every generated token.
