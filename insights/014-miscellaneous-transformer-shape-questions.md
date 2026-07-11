---
number: 14
title: "Miscellaneous Transformer Shape Questions"
book: "Inference Engineering"
tags: ["position-encoding","attention-heads","hidden-dimensions","unembedding","logits","transformer"]
---

# Insight 14: Miscellaneous Transformer Shape Questions

### Key Idea
Three useful shape-level clarifications are: position information is incorporated into token representations, attention temporarily uses smaller per-head vectors without permanently shrinking the hidden stream, and the unembedding matrix maps the final hidden vector into one logit per vocabulary token.

### Why It Works
After token lookup, position information is incorporated so the model knows where each token appears. Depending on the architecture, this may use learned positional embeddings, fixed mathematical encodings, or a method such as rotary position encoding. The durable mental model is that the token's starting representation is modified by position before or during attention.

Inside attention, the main hidden vector remains the model's primary representation. Each attention head projects that hidden vector into smaller temporary query, key, and value spaces. Those smaller vectors are used to compute attention. The outputs from many heads are then combined and projected back into the model's main hidden dimension. The hidden stream is therefore not permanently compressed.

At the end of the transformer, the final hidden vector for the last sequence position is multiplied by the unembedding matrix. If the hidden size is `d_model` and the vocabulary size is `vocab_size`, then:

```text
final hidden vector: 1 × d_model
unembedding matrix: d_model × vocab_size
resulting logits: 1 × vocab_size
```

The result contains one raw score for every token in the vocabulary.

### When It Matters
These distinctions matter when reasoning about tensor shapes and component ownership.

Position information changes how the model represents where a token appears.

Attention heads use smaller temporary working spaces, but the transformer keeps returning to the main hidden-vector size between blocks.

The unembedding matrix is the bridge from the transformer's continuous hidden space back to the model's fixed vocabulary.

### Mental Model
> Position tells a token where it is. Attention temporarily opens smaller workspaces. Unembedding turns the final hidden state into a vocabulary-sized scoreboard.

### Takeaway
The token representation begins in the model's hidden dimension, receives position information, and stays in that main dimension throughout the transformer stack. Attention uses smaller temporary per-head vectors and then combines them back into the hidden stream. Finally, the unembedding matrix converts the last hidden vector into one logit for every vocabulary token.
