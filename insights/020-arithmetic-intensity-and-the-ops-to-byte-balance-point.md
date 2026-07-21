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

This `295 operations/byte` is the hardware's approximate balance point. It says that, to keep compute and memory equally busy, a workload would need to perform about 295 floating-point operations for every byte it moves from memory.

The workload has its own arithmetic intensity. Compare that workload number with the hardware balance point.

```text
workload intensity < 295 operations/byte
→ memory-bound
```

For example:

```text
40 operations/byte < 295 operations/byte
```

The workload performs relatively little math for each byte moved. The GPU still has unused compute capacity, but performance is limited by how quickly memory can deliver data.

```text
workload intensity > 295 operations/byte
→ compute-bound
```

For example:

```text
500 operations/byte > 295 operations/byte
```

The workload asks for a large amount of computation for each byte moved. Memory can supply data fast enough, but the compute units become the limiting resource.

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

The hardware balance point is the dividing line. The workload's arithmetic intensity tells you which side of that line the workload falls on.

### Takeaway
The H100 example has an approximate hardware balance point of 295 floating-point operations per byte. A workload below that value is generally memory-bound; a workload above it is generally compute-bound. Arithmetic intensity belongs to the workload, while the 295 operations-per-byte threshold comes from the GPU's hardware capabilities.
