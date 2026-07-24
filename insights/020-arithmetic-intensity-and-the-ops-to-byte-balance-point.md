---
number: 20
title: "Arithmetic Intensity and the Ops-to-Byte Balance Point"
book: "Inference Engineering"
tags: ["arithmetic-intensity","ops-to-byte","memory-bound","compute-bound","gpu","h100"]
---

# Insight 20: Arithmetic Intensity and the Ops-to-Byte Balance Point

### Key Idea
Arithmetic intensity is the amount of mathematical work a workload performs per byte of memory traffic.

```text
workload arithmetic intensity = floating-point operations / bytes moved
```

A GPU also has a hardware balance point, found by dividing peak compute throughput by peak memory bandwidth.

### Why It Works
Using the H100 example:

```text
peak compute:          989 trillion floating-point operations per second
peak memory bandwidth: 3.35 trillion bytes per second
```

Divide the two rates:

```text
989 trillion operations/second
÷ 3.35 trillion bytes/second
≈ 295 operations/byte
```

The seconds cancel, leaving operations per byte.

This `295 operations/byte` is the hardware's approximate balance point. It is not something the hardware "gives" to each byte. It is the dividing line where compute speed and memory speed would both be fully used.

The workload has its own arithmetic intensity. Compare that workload number with the hardware balance point.

```text
workload intensity < 295 operations/byte
→ memory-bound
```

For example:

```text
40 operations/byte < 295 operations/byte
```

The workload performs relatively little math for each byte moved. Each byte creates too little work, so the compute units finish quickly and wait for more data. Performance is limited by how quickly memory can deliver bytes.

```text
workload intensity > 295 operations/byte
→ compute-bound
```

For example:

```text
500 operations/byte > 295 operations/byte
```

The workload asks for a large amount of computation for each byte moved. For every byte, the algorithm needs more operations than the hardware balance point can support while keeping memory and compute equally busy. The memory system can deliver enough data, but the compute units cannot finish the required operations fast enough.

### When It Matters
This matters when deciding which optimization can improve performance.

If a workload is memory-bound, adding more compute will not help much. Improvements must reduce memory traffic, increase data reuse, or provide more memory bandwidth.

If a workload is compute-bound, faster memory will not help much. Improvements must reduce the number of operations or provide more compute throughput.

This also explains why batching can help LLM decode. Batching lets the same loaded model weights participate in more operations, increasing the workload's arithmetic intensity and moving it closer to the compute-bound side.

### Mental Model
> Ask: what resource would need to improve for the workload to get faster?

```text
Memory-bound:
"The compute units are ready, but data is not arriving fast enough."

Compute-bound:
"The data is available, but the compute units cannot finish the required math fast enough."
```

Another way to say it:

```text
If the algorithm needs 500 operations per byte,
and the hardware balance point is 295 operations per byte,
then the workload is compute-bound.
```

Why? Because for every byte moved, the algorithm demands a lot of math. Memory can supply the data, but compute cannot satisfy the required operations quickly enough.

### Takeaway
The H100 example has an approximate hardware balance point of 295 floating-point operations per byte. A workload below that value is generally memory-bound; a workload above it is generally compute-bound. Arithmetic intensity belongs to the workload, while the 295 operations-per-byte threshold comes from the GPU's hardware capabilities.
