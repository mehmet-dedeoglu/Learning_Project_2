## Information
This repository is anonymous at the moment as this work is under review.

## Requirements
Please change the [name] and [prefix] in environment.yml in accordance with your system. Then, run the following code sequence for conda environment.
```setup
conda env create -f environment.yml
conda activate [name]
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

## Training
[figure_setup.txt] file contains all the required information for plotting results. First row contains xlabel, ylabel and title. Second line contains the list of legend texts split by `;'. Third row contains the paths to x axis tick values. Forth row contains the paths to y values. 

Before starting to train a model, please run the following code for any dataset, which will download datasets and arrange them as required:
```train
python Converter.py
```
More description will be provided...







