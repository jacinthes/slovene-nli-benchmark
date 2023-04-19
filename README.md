# Slovene NLI Benchmark Submission
This repository contains the submission to the [Slovene NLI Benchmark](https://slobench.cjvt.si/leaderboard/view/9).
The goal of this challenge is to improve the existing benchmark score for Slovene NLI, which measures the ability of a machine learning model to understand natural language inference in Slovene.

## Submitted model
The best performing model on the SI-NLI dataset is made available on HugginFace -> [cross-encoder-si-nli-snli-mnli](https://huggingface.co/jacinthes/cross-encoder-sloberta-si-nli-snli-mnli/)<br />
Best validation accuracy: **0.7751**
This model's predictions are submitted for evaluation.

A second model [cross-encoder-si-nli](https://huggingface.co/jacinthes/cross-encoder-sloberta-si-nli/) which was trained only on the SI-NLI training set is also available.<br />
Best validation accuracy: **0.7660**

## Repo structure
Folder *training* contains the training script which was used to train all the models during experimentation.<br />
Folder *inference* contains the notebook which can be used to make predictions using either of the two published models.<br />
Folder *translation* contains the script which was used to translate training samples.<br />
Section *Approach* describes all the steps taken during the development of NLI models.

# Approach
## Model architecture
[SentenceTransformers](https://arxiv.org/pdf/1908.10084.pdf) CrossEncoder class was used to train the models. The CrossEncoder uses a Siamese BERT-Network to encode both input sentences simultaneously (single pass) and then predicts the target value. It achieves state-of-the-art results on various sentence pair classification tasks - including NLI.

## Base model
[SloBERTa](https://huggingface.co/EMBEDDIA/sloberta) was used as the base model as it outperformed other models during experimentation ([xlm-roberta-large](https://huggingface.co/xlm-roberta-large), [CroSloEngual BERT](https://huggingface.co/EMBEDDIA/crosloengual-bert)). 

## Experimentation
1. Using only SI-NLI dataset to train the model
2. Extending the training set with translated English NLI samples
3. Extending the training set with synthetic samples
4. Extending the training set with data augmentation techniques

### 1. Using only SI-NLI dataset to train the model
During this step the model was trained using only SI-NLI training samples. The goal was to establish if using the crossencoder architecture will positively impact accuracy. The model was trained using the following hyperparameters:<br />
<br />
**Batch size: 24** -> yielded best results for this training set<br />
**Epochs: 15**<br />
**Sequence length: 102** -> recommended by the authors of the current highest benchmark score as the 99th percentile of the lengths in the training set<br />
**Loss function: CrossEntropyLoss**<br />
**Optimizer: AdamW**<br />
**Learning rate: 2e-5**<br />
**Warmup steps: 0.1** -> percent of training steps<br />
**Weight decay: 0.01** -> decided based on experimentation with the starting point calculated using the formula proposed in ([Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101))<br /> $w = λ \sqrt{b/BT}$<br />
where $b$ is the batch size, $B$ the number of training samples, $T$ the number of epochs and $λ$ a hyperparameter recommended by the authors to be $[0.025, 0.05]$
<br />
Dev set accuracy was used for model selection.<br />
The best model achieved the following dev set metrics:<br />
|               | precision | recall | f1     | support |
|---------------|-----------|--------|--------|---------|
| entailment    | 0.7409    | 0.7772 | 0.7586 | 184     |
| neutral       | 0.8092    | 0.7447 | 0.7756 | 188     |
| contradiction | 0.7514    | 0.7771 | 0.7640 | 175     |
|               |           |        |        |         |
| accuracy      | 0.7660    |        |        |         |

### 2. Extending the training set by translating existing English NLI datasets
The hypothesis was that translating high quality data points will be beneficial for the Slovene NLI model. To test this hypothesis, 40000 training examples were translated from two commonly used NLI datasets [SNLI](https://nlp.stanford.edu/projects/snli/) and [MNLI](https://cims.nyu.edu/~sbowman/multinli/). 20000 premise, hypothesis pairs from each dataset were translated using the Google translator. The exact translation code is provided in the folder translation. Other free translation options were tested [m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) but performed worse.<br />
The translated dataset is made available on [HuggingFace](https://huggingface.co/datasets/jacinthes/slovene_mnli_snli)
<br />
The model was trained using the following hyperparameters:<br />
**Batch size: 64** <br />
**Epochs: 8**<br />
**Sequence length: 102** -> the 99th percentile of the training set lengths is 88, but to avoid truncating SI-NLI training samples, 102 was used again.<br />
**Loss function: CrossEntropyLoss**<br />
**Optimizer: AdamW**<br />
**Learning rate: 2e-5**<br />
**Warmup steps: 0.1** -> <br />
**Weight decay: 0.002** <br />
To maximize the benchmark score, SI-NLI dev set accuracy was used for model selection. Although the combined si-nli, snli and mnli dev set is also available on Huggingface. <br />
The best model achieved the following SI-NLI dev set metrics:<br />
|               | precision | recall | f1     | support |
|---------------|-----------|--------|--------|---------|
| entailment    | 0.8079    | 0.7409 | 0.7730 | 193     |
| neutral       | 0.7254    | 0.8092 | 0.7650 | 173     |
| contradiction | 0.7966    | 0.7790 | 0.7877 | 181     |
|               |           |        |        |         |
| accuracy      | 0.7751    |        |        |         |

### 3. Extending the training set with synthetic samples
Generative AI has also become a very powerful tool for generating synthetic labeled data for low resource domains. I use OpenAI's davinci003 model (GPT3.5) to generate 3000 training samples. GPT is instructed to generate 3 hypotheses for each given premise - one for each NLI label. The list of premises is a list of 1000 sentences from the [Gigafida](https://huggingface.co/datasets/cjvt/cc_gigafida) corpus. Sentences containing "?" were removed. The generation script, GPT prompt, sentences and the generated dataset are all available in the *GPT3.5 synthetic data generation* folder.
<br />
The model was trained using the following hyperparameters:<br />
**Batch size: 32** <br />
**Epochs: 10**<br />
**Sequence length: 102**<br />
**Loss function: CrossEntropyLoss**<br />
**Optimizer: AdamW**<br />
**Learning rate: 2e-5**<br />
**Warmup steps: 0.1** -> <br />
**Weight decay: 0.01** <br />
<br />
The model performed worse than the above options and achieved the highest accuracy of: **0.7239**<br />
Upon further inspection of the training data, GPT struggles with differentiating between neutral and entailment labels. A large percent of generated neutral hypotheses should be classified as entailment. Nonetheless, GPT shows promise for generating training data for low resource domains/languages.
