---
number: 1
title: "Latency vs. Throughput"
book: "Inference Engineering"
tags: ["latency","throughput","batching","gpu","cost"]
---

# Insight 1: Latency vs. Throughput

### Key Idea
Latency is the end-to-end time from when a request arrives until the response is returned. Throughput is the number of requests the system completes per second.

### Why It Works
The latency-throughput tradeoff comes from batching. By waiting briefly to collect multiple requests, the system can process them together in one larger, more efficient GPU operation. This improves throughput because the GPU does fewer separate kernel launches, uses memory more efficiently, and performs larger matrix operations.

The model itself is not slower. The system intentionally adds a small amount of queueing delay so the GPU can do more useful work per unit of time.

### When It Matters
Low latency matters for interactive applications like chatbots, autocomplete, and real-time APIs where users notice responsiveness.

High throughput matters for offline or batch jobs because requests can wait longer. In those settings, batching is cheaper because the same GPU can process more total work, reducing the number of GPUs needed.

### Mental Model
> Trade a small amount of queueing delay for much more efficient GPU execution.

### Takeaway
Latency vs. throughput is mainly a cost vs. responsiveness tradeoff. Optimizing for throughput often makes each individual request wait a bit longer, but it can make the system much cheaper at scale.
