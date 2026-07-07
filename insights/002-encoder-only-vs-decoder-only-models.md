---
number: 2
title: "Encoder-Only vs. Decoder-Only Models"
book: "Inference Engineering"
tags: ["encoder","decoder","bert","embeddings","llm-architecture"]
---

# Insight 2: Encoder-Only vs. Decoder-Only Models

### Key Idea
An encoder produces contextual embeddings, not generated text. A decoder uses its internal representations to predict and generate the next token.

### Why It Works
Encoder models, such as BERT, allow every token to attend to every other token in the input. This creates rich contextual representations of the input sequence. The output is usually a set of semantic vectors, either one vector per token or one pooled vector for the whole sequence.

Those vectors are then used by a task-specific head for things like classification, question answering, semantic search, retrieval, or token labeling.

Decoder models also build internal representations, but they use those representations to predict the next token autoregressively. Modern chat LLMs are often decoder-only because the same architecture can both read the prompt and generate the response.

### When It Matters
Encoder-only models are useful when the job is to understand, classify, rank, embed, or retrieve information.

Decoder-only models are useful when the job is to generate text, code, summaries, translations, or chat responses.

### Mental Model
> An encoder is a feature extractor; a decoder is a writer.

### Takeaway
Decoder-only does not mean the model skips understanding the input. It still builds internal semantic representations of the prompt. It simply means there is no separate encoder network; the decoder itself does the understanding needed to generate the next token.
