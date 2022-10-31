# Predicting Effective Arguments

A model which classifies argumentative elements such as Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and Concluding Statement as "effective," "adequate," or "ineffective." based on essays written by U.S. students in grades 6-12.

# Installation and Dependencies:
We recommend using conda environment to install dependencies of this library first. Please install (or load) conda and then proceed with the following commands:
```
conda create -n dl_project python=3.9
conda activate dl_project
conda install pytorch torchvision torchaudio -c pytorch
conda install -c pytorch torchtext
```

Sometimes, there might be some errors when using torchtext such as
```
ModuleNotFoundError: No module named 'torchtext.legacy'
```

In this case, try downgrade your torchtext 
```
pip install torchtext==0.10.0
```




