# Slovene NLI Benchmark Submission
This repository contains my submission to the Slovene NLI Benchmark. 
The goal of this challenge is to improve the existing benchmark score for Slovene NLI, which measures the ability of a machine learning model to understand natural language inference in Slovene.

## Approach
### Model architecture
[SentenceTransformers](https://arxiv.org/pdf/1908.10084.pdf) CrossEncoder class was used to train the models. The CrossEncoder uses a Siamese BERT-Network to encode both input sentences simultaneously (single pass) and then predicts the target value. It achieves state-of-the-art results on various sentence pair classification tasks - including NLI.

###
