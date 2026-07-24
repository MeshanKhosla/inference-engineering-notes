---
number: 21
title: "Standard Attention Read Compute Write"
book: "Inference Engineering"
tags: ["attention","arithmetic-intensity","hbm","memory-traffic","softmax","gpu"]
---

# Insight 21: Standard Attention Read Compute Write

### Key Idea
Standard attention can be understood as three read-compute-write steps. Each step reads matrices from GPU memory, performs a calculation, and writes the result back to memory. Arithmetic intensity comes from comparing the total compute work against the total memory traffic.

### Why It Works
For one attention head, the main matrices are:

```text
N = sequence length, or number of tokens
d = attention head dimension

Q = query matrix, shape N × d
K = key matrix, shape N × d
V = value matrix, shape N × d

S = score matrix, shape N × N
P = probability / attention-weight matrix, shape N × N
O = output matrix, shape N × d
```

The standard attention flow is:

```text
S = QKᵀ
P = softmax(S)
O = PV
```

In words:

```text
queries compare to keys → raw attention scores
scores pass through softmax → attention weights
attention weights mix values → output vectors
```

### Read, Compute, Write Pattern
Assume FP16 inference, where each matrix value is `2 bytes`.

#### Step 1: Compute Attention Scores
Read:

```text
Q = 2 × N × d bytes
K = 2 × N × d bytes
```

Compute:

```text
S = QKᵀ
```

Shape:

```text
Q:  N × d
Kᵀ: d × N
S:  N × N
```

Each score is a dot product between a query and a key:

```text
S[i][j] = dotProduct(Q[i], K[j])
```

Write:

```text
S = 2 × N × N bytes
```

#### Step 2: Compute Attention Weights
Read:

```text
S = 2 × N × N bytes
```

Compute:

```text
P = softmax(S)
```

Softmax is applied row by row. Each row becomes the attention-weight distribution for one token.

Write:

```text
P = 2 × N × N bytes
```

#### Step 3: Compute Attention Output
Read:

```text
P = 2 × N × N bytes
V = 2 × N × d bytes
```

Compute:

```text
O = PV
```

Shape:

```text
P: N × N
V: N × d
O: N × d
```

Each output vector is a weighted sum of value vectors.

Write:

```text
O = 2 × N × d bytes
```

### Connecting to Arithmetic Intensity
Arithmetic intensity is:

```text
work / memory traffic
```

For standard attention, the work comes from computing:

```text
S = QKᵀ
P = softmax(S)
O = PV
```

The memory traffic comes from reading and writing the matrices involved:

```text
reads:  Q, K, S, P, V
writes: S, P, O
```

Using the simple FP16 accounting above, the total memory traffic is the sum of all reads and writes:

```text
read traffic:
Q + K + S + P + V
= 2Nd + 2Nd + 2N² + 2N² + 2Nd
= 6Nd + 4N²

write traffic:
S + P + O
= 2N² + 2N² + 2Nd
= 2Nd + 4N²

total memory traffic:
8Nd + 8N² bytes
```

The important systems point is that standard attention materializes the large `N × N` matrices `S` and `P` in HBM. Writing them to GPU memory and reading them back creates a lot of memory traffic.

### When It Matters
This matters when connecting attention to GPU bottlenecks. If attention moves a lot of data relative to the amount of math it performs, its arithmetic intensity may fall below the hardware's ops-to-byte balance point, making it memory-bound.

It also sets up why optimized attention kernels try to avoid unnecessary HBM traffic. If an implementation can compute attention without writing huge intermediate matrices like `S` and `P` to HBM, it can reduce memory traffic and increase arithmetic intensity.

### Mental Model
> Standard attention is not just math. It is also a memory movement pattern.

```text
Read Q and K → compute S → write S
Read S → compute P → write P
Read P and V → compute O → write O
```

The more often large intermediate matrices are written to and read from HBM, the more memory traffic the algorithm creates.

### Takeaway
Standard attention computes `S = QKᵀ`, `P = softmax(S)`, and `O = PV`. The arithmetic-intensity calculation counts both the compute work and the memory traffic from reading and writing `Q`, `K`, `V`, `S`, `P`, and `O`. The large `N × N` score and probability matrices are especially important because they create substantial HBM traffic.
