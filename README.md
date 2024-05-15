# Learning Effective Representations for Retrieval using Self-Distillation with Adaptive Relevance Margins

--------

## Overview

This is the code repository for the paper "Learning Effective Representations for Retrieval using Self-Distillation with Adaptive Relevance Margins".

> Representation-based retrieval models, so-called bi-encoders, estimate the relevance of a document to a query by calculating the similarity of their respective embeddings. Current state-of-the-art bi-encoders are trained using an expensive training regime involving knowledge distillation from a teacher model and extensive batch-sampling techniques. Instead of relying on a teacher model, we contribute a novel parameter-free loss function for self-supervision that exploits the pre-trained text similarity capabilities of the encoder model as a training signal, eliminating the need for batch sampling by performing implicit hard negative mining. We explore the capabilities of our proposed approach through extensive ablation studies, demonstrating that self-distillation can match the effectiveness of teacher-distillation approaches while requiring only a fraction of the data and compute.

Supplementary data (TREC-format run files for all final trained models) is [hosted on Zenodo](https://zenodo.org/records/11197962).

## Project Organization

```
├── Dockerfile         <- Dockerfile with all dependencies for reproducible execution
├── LICENSE            <- License file
├── Makefile           <- Makefile with commands to reproduce artifacts (data + models)
├── README.md          <- The top-level README for project
├── configs            <- Configuration files for model and sweep parameters
├── data               <- Data folder; will be populated by data scripts
├── main.py            <- Main Lightning CLI entrypoint
├── requirements.txt   <- Dependencies
├── scripts            <- Scripts to automate single tasks (data parsing, sweep agents, ...)
├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
└── src                <- Model source code
```

## Replication

Data, model training, and evaluation is replicable with `make` targets:

```
$ make
Available rules:

requirements        Install Python Dependencies 
data-train          Download and preprocess train dataset 
data-eval           Download and preprocess eval datasets
fit                 Run the training process
eval                Run eval process 
clean               Delete all compiled Python files 

```

These can be run in the given order to fully replicate the experimental pipeline. 

Each training run from the paper can be executed with its given config file in `configs` with the following command:
```sh
python3 main.py fit -c <path-to-config-file>
```


--------
