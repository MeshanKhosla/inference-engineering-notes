---
number: 19
title: "Batching Amortizes Model-Weight Traffic"
book: "Inference Engineering"
tags: ["batching","decode","memory-bandwidth","kv-cache","throughput","gpu"]
---

# Insight 19: Batching Amortizes Model-Weight Traffic

### Key Idea
During decode, batching improves throughput mainly by amortizing shared model-weight reads across multiple requests. Each request still has its own KV cache, so KV-cache traffic continues to scale with the number of requests and their context lengths.

### Why It Works
For one decode token, the GPU must use the model's learned weights, including attention matrices such as `W_Q`, `W_K`, `W_V`, and `W_O`, as well as the MLP and other layer weights.

Without batching, processing `N` requests separately means the same shared model weights are read and used in `N` separate decode passes.

With batching, the GPU can load a weight matrix and apply it to a batch of token vectors together. This performs more computation for roughly the same shared model-weight traffic.

However, batching does not eliminate KV-cache traffic. Every request has its own cached keys and values from its own previous tokens, and those caches still need to be read during attention.

A useful approximation is:

```text
total decode memory traffic
≈ shared model-weight traffic
+ per-request KV-cache traffic
```

For a batch of `N` requests:

```text
without batching:
load shared model weights N separate times
+ read N separate KV caches

with batching:
load shared model weights roughly once for the batch
+ still read N separate KV caches
```

The total memory traffic is therefore not constant. What improves is the amount of useful computation performed per byte of shared model-weight data moved.

### When It Matters
Decode is often memory-bound because generating one token requires reading large model weights while performing relatively little computation.

Batching increases arithmetic intensity:

```text
arithmetic intensity = computation / memory traffic
```

A larger batch lets each loaded model weight participate in more multiplications. This makes better use of the GPU and improves throughput.

At large batch sizes or long context lengths, KV-cache bandwidth and capacity can become important bottlenecks again because each request contributes its own cache traffic.

### Mental Model
> Bring the shared cookbook out once and use it to cook many meals. Each meal still needs its own ingredients.

The cookbook is the model weights. The separate ingredients are the requests' KV caches.

### Takeaway
Batching does not make decode memory traffic constant and does not remove KV-cache reads. It mainly reduces model-weight traffic per generated token by reusing the same loaded weights across many requests in the batch.
