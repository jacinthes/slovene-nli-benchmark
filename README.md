# Slovene NLI Benchmark Submission
This repository contains my submission to the [Slovene NLI Benchmark](https://slobench.cjvt.si/leaderboard/view/9).
The goal of this challenge is to improve the existing benchmark score for Slovene NLI, which measures the ability of a machine learning model to understand natural language inference in Slovene.

## Approach
### Model architecture
[SentenceTransformers](https://arxiv.org/pdf/1908.10084.pdf) CrossEncoder class was used to train the models. The CrossEncoder uses a Siamese BERT-Network to encode both input sentences simultaneously (single pass) and then predicts the target value. It achieves state-of-the-art results on various sentence pair classification tasks - including NLI.

### Experimentation
1. Using only SI-NLI dataset to train the model
2. Extending the training set with translated English NLI samples
3. Extending the training set with synthetic samples
4. Extending the training set with data augmentation techniques

##cross-encoder-sloberta-si-nli

