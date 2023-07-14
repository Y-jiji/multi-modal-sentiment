# Lab 5

This is the repository for contemporary artificial intelligence course lab assignment in ECNU. 

## Environment

The dependencies for this project is maintained with [micromamba](https://github.com/mamba-org/mamba), an alternative to conda with a faster built-in SAT solver. 

To install the dependencies, you may run the following command: 

```shell
conda env create -f environment.yml
```

This command will create a conda/micromambda environment called `ds` (data science). Don't forget to activate it with: 

```bash
conda activate ds
```

## Project Layout

```bash
..
├───.gitignore
├───answer.csv			# output for evaluation
├───config.py			# general configuration
├───data.py				# data preprocessing
├───LICENSE
├───main.py				# entry point
├───README.md
├───data				# data folder for big files
├───meta				# data folder for csv tables
└───zoo 				# models
    ├───attenagg.py
    ├───blip.py
    └───postagg.py
```

## To Run

I'm not a fan of putting parameters into command line arguments. 

Therefore, you are only allowed to reproduce the experimental results with the following command: 

To run the text + image model, use: 

```shell
python main.py
```

To run the ablation study, use: 

```shell
python main.py --only-text
python main.py --only-image
```

To generate the answer.csv, use: 

```shell
python main.py --get-output
```

## Architecture

The model architecture is depicted in the following picture, which is not elegant in general, but it works well. 

![图片1](./assets/%E5%9B%BE%E7%89%871.png)

## Attribution

[BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

[ResNet](https://pytorch.org/hub/pytorch_vision_resnet)

