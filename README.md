# Slovene NLI Benchmark Submission
This repository contains the submission to the [Slovene NLI Benchmark](https://slobench.cjvt.si/leaderboard/view/9),  a challenge aimed at improving the benchmark score for Slovene natural language inference (NLI) models. This submission achieves new state of the art results on the Slovene NLI Benchmark.

## Published models
I have made available the best performing model through Hugging Face -> [cross-encoder-sloberta-si-nli-snli-mnli](https://huggingface.co/jacinthes/cross-encoder-sloberta-si-nli-snli-mnli/)<br />
This model achieved the following metrics:
| Test accuracy | 77.15 |
|---------------|-------|
| Dev accuracy  | 77.51 |

I have also published a second model [cross-encoder-sloberta-si-nli](https://huggingface.co/jacinthes/cross-encoder-sloberta-si-nli/), which was trained only on the SI-NLI training data, achieving:
| Test accuracy | 75.95 |
|---------------|-------|
| Dev accuracy  | 75.14 |

## Repo structure
Folder *training* contains the training script, which was used to train all the models during experimentation.<br />
Folder *inference* contains the notebook, which can be used to make predictions using either of the two published models.<br />
Folder *translation* contains the script, which was used to translate training samples.<br />
Folder *GPT3.5 synthetic data generation* contains synthetic data generator script, GPT prompt, sentence list and the generated dataset.<br />
Folder *back translations* contains the dataset generated with the back translation technique.
Section *Approach* describes all the steps taken during the development.

# Approach
## Model architecture
[SentenceTransformers](https://arxiv.org/pdf/1908.10084.pdf) [CrossEncoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) class was used to train the models. The CrossEncoder uses a Siamese BERT-Network to encode both input sentences simultaneously (single pass) and then predicts the target value. It achieves state-of-the-art results on various sentence pair classification tasks - including NLI.

## Base model
[SloBERTa](https://huggingface.co/EMBEDDIA/sloberta) was used as the base model as it outperformed other models during experimentation ([xlm-roberta-large](https://huggingface.co/xlm-roberta-large), [CroSloEngual BERT](https://huggingface.co/EMBEDDIA/crosloengual-bert)). 

## Experimentation
I conducted a series of experiments to improve model's performance:
1. Using only the SI-NLI dataset to train the model.
2. Extending the training set with translated English NLI samples.
3. Extending the training set with synthetic samples.
4. Extending the training set with data augmentation techniques.

### 1. Using only the SI-NLI dataset to train the model
To evaluate the impact of using the CrossEncoder architecture on model accuracy, I trained the model using only the SI-NLI training set. The following hyperparameters were used for this experiment:<br />
<br />

| Hyperparameter    | Value            | Reason                                                                                    |
|-------------------|------------------|-------------------------------------------------------------------------------------------|
| Batch size        | 64               | Yielded better results compared to 16, 32, 48 and 96                                      |
| Epochs            | 15               |                                                                                           |
| Sequence length   | 102              | Same as the current benchmark leader: 99th percentile of the lengths in the training set |
| Loss function     | CrossEntropyLoss |                                                                                           |
| Optimizer         | AdamW            |                                                                                           |
| Learning rate     | 2e-5             |                                                                                           |
| Warmup steps      | 0.1              | Percent of all training steps.                                                            |
| Weight decay rate | 0.0015             | Decided based on experimentation with the starting point calculated using the formula proposed in  ([Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101))  $w = λ \sqrt{b/BT}$ where $b$ is the batch size,  $B$ the number of training samples,  $T$ the number of epochs and $λ$ a hyperparameter recommend by the authors to be in $[0.025, 0.05]$                                                                                     |


Dev set accuracy was used for model selection.<br />
The best model achieved the following dev set metrics:<br />
|               | precision | recall | f1     | support |
|---------------|-----------|--------|--------|---------|
| entailment    | 0.7740    | 0.7098 | 0.7405 | 193     |
| neutral       | 0.7150    | 0.7977 | 0.7541 | 173     |
| contradiction | 0.7684    | 0.7514 | 0.7598 | 181     |
|               |           |        |        |         |
| accuracy      | 0.7514    |        |        |         |

### 2. Extending the training set by translating existing English NLI datasets
To test the hypothesis that translating high-quality data points will be beneficial for the Slovene NLI model, 40000 training examples were translated from two commonly used NLI datasets [SNLI](https://nlp.stanford.edu/projects/snli/) and [MNLI](https://cims.nyu.edu/~sbowman/multinli/). 20000 premise hypothesis pairs from each dataset were translated using the Google translator. The exact translation is provided in the folder *translation*. Other free translation options were tested [m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) but performed worse.<br />
The translated dataset is made available on [HuggingFace](https://huggingface.co/datasets/jacinthes/slovene_mnli_snli)
<br />

| Hyperparameter    | Value            | Reason                                                                                    |
|-------------------|------------------|-------------------------------------------------------------------------------------------|
| Batch size        | 64               | Yielded better results compared to 16, 32, 48 and 96                                      |
| Epochs            | 8                | The model converged faster trained on a larger dataset                                    |
| Sequence length   | 102              | the 99th percentile of the training set lengths is 88, but to avoid truncating the SI-NLI training samples, 102 was used again. |
| Loss function     | CrossEntropyLoss |                                                                                           |
| Optimizer         | AdamW            |                                                                                           |
| Learning rate     | 2e-5             |                                                                                           |
| Warmup steps      | 0.1              |                                                                                           |
| Weight decay rate | 0.002            |                                                                                           |

To maximize the benchmark score, the SI-NLI dev set accuracy was used for model selection. Although the combined si-nli, snli and mnli dev sets are also available on HuggingFace. <br />
The best model achieved the following SI-NLI dev set metrics:<br />
|               | precision | recall | f1     | support |
|---------------|-----------|--------|--------|---------|
| entailment    | 0.8079    | 0.7409 | 0.7730 | 193     |
| neutral       | 0.7254    | 0.8092 | 0.7650 | 173     |
| contradiction | 0.7966    | 0.7790 | 0.7877 | 181     |
|               |           |        |        |         |
| accuracy      | 0.7751    |        |        |         |

### 3. Extending the training set with synthetic samples
Generative AI has also a very powerful tool for generating synthetic labeled data for low resource domains. I use OpenAI's davinci003 model (GPT3.5) to generate 3000 training samples. GPT is instructed to generate 3 hypotheses for each given premise - one for each NLI label. The list of premises is a list of 1000 sentences from the [Gigafida](https://huggingface.co/datasets/cjvt/cc_gigafida) corpus. Sentences containing "?" were removed. The generation script, GPT prompt, sentences and the generated dataset are all available in the *GPT3.5 synthetic data generation* folder.
<br />
<br />
The model was trained using the same hyperparameters as the model from 1.<br />
It performed worse than the above options and achieved the accuracy of: **72.39**<br />
<br />
Upon further inspection of the training data, GPT struggles with differentiating between neutral and entailment labels. A large percent of generated neutral hypotheses should be classified as entailment. Nonetheless, GPT shows promise for generating training data for low resource domains/languages.

### 4. Extending the training set with data augmentation techniques
Augmenting the original dataset with various techniques can lead to better model performance. I experiment with back translation. The original premise hypothesis pair is first translated to English and then back to Slovene. This did lead to slight variation in sentence wording while preserving the meaning. The generated dataset is available in the folder *back translation data*.
<br />
<br />
The model was trained using the same hyperparameters as the model from 1.<br />
The model performed slightly worse than the 1. and 2. options and achieved the accuracy of: **74.4**<br />
<br />
While this experiment did not negatively impact performance on the SI-NLI dev set, the generalization benefits of this technique are uncertain. While back translating the SI-NLI data did expand the training set, it does not necessarily guarantee improved generalization to new, unseen data.
