---
number: 3
title: "Token Vocabulary"
book: "Inference Engineering"
tags: ["tokens","vocabulary","tokenizer","embeddings","llm-architecture"]
---

# Insight 3: Token Vocabulary

### Key Idea
A model's vocabulary is the fixed set of tokens that its tokenizer knows how to produce. It is not every possible word or string in a language.

### Why It Works
The tokenizer decomposes arbitrary text into sequences of tokens from this fixed vocabulary. Since tokens are composable, a vocabulary of around 100,000 tokens is enough to represent virtually any text, including words the model has never seen before.

The model only knows how to process and generate tokens from this fixed vocabulary. Full words, rare terms, names, and new phrases can still be represented by combining multiple smaller tokens.

### Example
Suppose a tokenizer's vocabulary contains the tokens:

- `"un"`
- `"believ"`
- `"able"`

The word:

`unbelievable`

can be represented as:

`"un" + "believ" + "able"`

Similarly, a new word like:

`ChatGPTification`

might be tokenized into:

`"Chat" + "GPT" + "ification"`

even if the full word is not itself a vocabulary token.

### Mental Model
> The vocabulary is the model's alphabet, not its dictionary.

Instead of 26 letters, an LLM has roughly 100,000 reusable token building blocks. Just as letters combine to form infinitely many words, tokens combine to represent virtually any text.

### Takeaway
The vocabulary defines all the pieces the model can recognize and generate. The model does not need a token for every possible word because it constructs language by composing tokens from its fixed vocabulary.
