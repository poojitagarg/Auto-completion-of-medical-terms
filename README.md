# Auto-completion of medical terms

### List of contents

- [Introduction](#introduction)
- [Working](#working)
- [Installation](#installation)

## Introduction
---
[(Back to top)](#list-of-contents)
To develop an AI tool to make the process of providing the electronic prescription a hassle free task for them. We aim to use an integrated algorithm which includes:

1. Recognizing handwriting employing a deep learning model.
2. Character level NLP model to complete the prescribed drug.

An input of a few letters will be enough for the algorithm to predict the whole word thus, saving time and effort of the doctor and increasing the efficiency. 

## Working
---
[(Back to top)](#list-of-contents)
# 1. Handwriting Recognition
- Dataset: MNIST, A-Z Kaggle Dataset
- Data Preprocessing: Resizing, Augmentation and Channel increasing
- Model: Pretrained Resnet50 Architecture with Imagenet weights
- Results:
![Imgur](https://i.imgur.com/HEI9C10.png)


## Architectural Diagram
![img]()
