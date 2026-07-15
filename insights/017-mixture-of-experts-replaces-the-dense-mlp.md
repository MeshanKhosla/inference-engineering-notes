---
number: 17
title: "Mixture of Experts Replaces the Dense MLP"
book: "Inference Engineering"
tags: ["mixture-of-experts","moe","router","experts","mlp","sparse-activation"]
---

# Insight 17: Mixture of Experts Replaces the Dense MLP

### Key Idea
Mixture of Experts replaces the single dense MLP graph in some transformer layers with many separate expert MLP graphs and a router. For each token, the router selects only a small number of experts to run.

### Why It Works
In a normal dense transformer layer, every token passes through the same full MLP:

```text
hidden vector
→ one dense MLP graph
→ updated hidden vector
```

In an MoE layer, that one graph is replaced by many separate MLP graphs:

```text
hidden vector
→ router
→ choose a few expert MLP graphs
→ combine their outputs
→ updated hidden vector
```

It is not one graph with fewer neuron-level paths. Each expert is its own feedforward neural network with its own learned weights.

For example, a layer might contain 64 expert MLPs, but the router may select only 2 experts for a particular token. The other 62 experts do not run for that token.

Routing usually happens per token. During prefill, different prompt tokens can be routed to different experts in parallel. During decode, the newest generated token is routed through selected experts after attention.

Experts are not normally assigned explicit human labels. Training may cause some experts to become more useful for patterns involving code, certain languages, mathematical structures, grammar, entities, or other features, but those specializations emerge from training.

### Example
```ts
function runMixtureOfExpertsForOneToken(
  hiddenVector: Vector,
  router: Router,
  experts: ExpertMlp[]
): Vector {
  // The router scores all available expert MLP graphs.
  const routingScores = router.scoreExperts(hiddenVector);

  // Only a small number of experts are selected for this token.
  const selectedExperts = chooseTopKExperts(routingScores, 2);

  const weightedExpertOutputs = selectedExperts.map(selection => {
    const expertOutput = experts[selection.expertIndex].forward(hiddenVector);

    return scale(expertOutput, selection.routingWeight);
  });

  const combinedExpertOutput = sum(weightedExpertOutputs);

  // Return to the same hidden-vector stream used by the transformer.
  return add(hiddenVector, combinedExpertOutput);
}
```

Conceptually:

```text
"fluffy" hidden vector
→ router
→ expert MLP 2 + expert MLP 7
→ combine outputs
→ updated "fluffy" hidden vector
```

Another token in the same prompt may be routed differently:

```text
"creature" hidden vector
→ router
→ expert MLP 4 + expert MLP 9
→ combine outputs
→ updated "creature" hidden vector
```

### When It Matters
MoE matters because it lets a model have many more total learned parameters without activating every parameter for every token.

```text
many total expert parameters
+
only a few active experts per token
=
more model capacity without proportional compute
```

Attention still runs before the MoE block. Attention lets tokens exchange information. The router then decides which expert MLP graphs should transform each token's context-aware hidden vector.

### Mental Model
> Dense MLP: every token goes through one full neural-network graph. MoE: many separate graphs exist, and the router chooses only a few for each token.

### Takeaway
Mixture of Experts does not choose a smaller path through one MLP. It replaces the dense MLP with many distinct expert MLPs. A router selects a small subset for each token, their outputs are combined, and the resulting vector continues through the transformer.
