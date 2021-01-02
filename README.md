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
3. Web Application to showcase the results

An input of a few letters will be enough for the algorithm to predict the whole word thus, saving time and effort of the doctor and increasing the efficiency. 

## Working
---
[(Back to top)](#list-of-contents)
### 1. Handwriting Recognition
- Dataset: MNIST, A-Z Kaggle Dataset
- Data Preprocessing: Resizing, Augmentation and Channel increasing
- Model: Pretrained Resnet50 Architecture with Imagenet weights
- Results:

![img](https://imgur.com/HEI9C10.png)

### 2. Character level NLP model
- Dataset: Medical Dictionary with 23500+ Prescription Drugs
- Preprocessing: Data Cleaning, One hot encoding 
- Models: Multiple models for multi length input.
- Accuracy: In range of 86-93%
- Model Architechture:

![img](https://imgur.com/fn9S9Tp.png)

- Model selection:

![img](https://imgur.com/gxzjhq5.png)

- Results: Using Previous algorithm and adding a 20% error prediction difference we get multiple outputs.

![img](https://imgur.com/c6SyTEn.png)

### 3. Web Application
>>GIF

## Installation
---
[(Back to top)](#list-of-contents)

These instructions assume you have `git` installed for working with Github from command window.

1. Clone the repository, and navigate to the downloaded folder. Follow below commands.
```
git clone https://github.com/poojitagarg/minor_project.git
cd web_app1

```

2. Creating python virtual environment using virtualenv package using following lines of code.

NOTE: For this step make sure you have virtualenv package installed.

```
virtualenv venv
source venv/bin/activate

```

3. Install few required pip packages, which are specified in the requirements.txt file .
```
pip3 install -r requirements.txt

```






