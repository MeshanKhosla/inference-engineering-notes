---
number: 15
title: "Multi-Head Attention and Learned Attention Patterns"
book: "Inference Engineering"
tags: ["multi-head-attention","attention-heads","training-vs-inference","attention-patterns","transformer"]
---

# Insight 15: Multi-Head Attention and Learned Attention Patterns

### Key Idea
Multi-head attention lets many attention heads run in parallel within a single transformer layer. Each head has its own learned query, key, and value matrices, allowing it to learn different kinds of token relationships.

### Why It Works
Inside one transformer layer, every attention head receives the same hidden vectors, but each head uses its own `W_Q`, `W_K`, and `W_V` matrices. Because these matrices are learned independently during training, different heads can specialize in different matching behaviors.

Training does **not** explicitly assign a job to each head. The only objective is to improve next-token prediction. Over many training updates, different heads often learn useful attention patterns.

After training, the learned matrices are fixed. During inference, each head computes a new attention pattern based on the current input.

### Example Patterns
Some heads may learn relationships that resemble:

- Adjective → noun
- Subject → verb
- Verb → subject
- Pronoun → referent
- Matching punctuation or quotation marks
- Repeated-token tracking
- Nearby token or phrase-boundary tracking

These are useful human interpretations. Internally, each head simply computes query/key similarities and attention weights.

### Mental Model
> Every attention head asks its own kind of question in parallel.

For example, one head might behave like:

```text
Which earlier words describe this noun?
```

while another might behave like:

```text
Which noun is the subject of this verb?
```

### When It Matters
For a GPT-3-style model with 96 layers and 96 attention heads per layer, there are 9,216 separate attention heads.

This does **not** mean there are exactly 9,216 fixed human-readable questions. Instead, there are 9,216 learned attention mechanisms. Some heads specialize, some overlap, some cooperate, and some are difficult to interpret.

### Takeaway
Training learns the matrices that define how each attention head performs matching. Inference uses those learned matrices to generate fresh, input-dependent attention patterns for every prompt. A head does not store one fixed attention map; it stores the learned machinery that produces attention maps.
