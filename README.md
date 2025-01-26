# Classification of Celestial Objects from SDSS DR14
Classification of Stars, Galaxies, and Quasars (QSO) from the Sloan Digital Sky Servey (SDSS) Data Release 14 (DR14)

# Project Overview
The aim of this project is to compare two different classification methods, a traditional machine learning (ML) technique and a Neural Network (NN), using real world data from SDSS DR14. Both methods will attempt to classify Stars, Galaxies, and Quasars (QSO), something that is essential in Astronomy. Often trying to distinquish between these objects can be difficult as they are all similar "point-like" objects on the night sky, so training a ML algorithm to be able to identify them is incredibly useful.

The project is structured into three main notebooks: 

## Question 1
This notebook focuses on explaining what a Decision Tree (DT) is and how it can be used to classify objects. It walks through the process of importing the dataset into a pandas dataframe, and then trimming the data ready for use. I then explain how to initialise the decision tree, train it, and test it to produce a classification report. Finally, I show some useful plots that can help visualise what the tree is doing, and how well it performs.

## Question 2
This notebook focuses on explaining what a Feed Forward Neural Network (FNN) is, how it can be trained to classify objects, and how successful it is with the SDSS DR14 dataset. I show step by step how the data needs to be preprocessed for use in the Neural Network (NN), and then show how the NN is defined, trained, and tested. Like in Question 1, I also show some useful plots that help visualise the model's performance. Finally, I compare the results of both the DT and the FNN.

## Question 3
This notebook explores how changing certain parts of the NN in Question 2 can affect the model's performance. I explain what happens if the ratio between the amount of training and testing data is changed, as well as how an inbalance between different classes can lead to over or underfitting. I discuss ways that the accuracy of the model can be improved, and why for this particular dataset, the accuracy was so high.

# Overview of the Dataset
The dataset used for this project comes from [Kaggle](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey/data).

To learn more about SDSS click [here](https://skyserver.sdss.org/dr14/en/home.aspx).

For SDSS's glossary click [here](https://live-sdss4org-dr14.pantheonsite.io/help/glossary).

The data contains 18 columns and 10,000 rows. Below are the column headings and what they represent:

| Column Name        | Data Type      | Description                                                                           |
|--------------------|----------------|---------------------------------------------------------------------------------------|
| objID              | float64        | Unique identifier for each astronomical object in the catalog                         |
| ra                 | float64        | Right Ascension (RA) of the object in decimal degrees                                 |
| dec                | float64        | Declination (DEC) of the object in decimal degrees                                    |
| u                  | float64        | Magnitude of the object in the SDSS U filter                                          |
| g                  | float64        | Magnitude of the object in the SDSS G filter                                          |
| r                  | float64        | Magnitude of the object in the SDSS R filter                                          |
| i                  | float64        | Magnitude of the object in the SDSS I filter                                          |
| z                  | float64        | Magnitude of the object in the SDSS Z filter                                          |
| run                | int64          | Run number of the observation                                                         |
| rerun              | int64          | Rerun number for calibration                                                          |
| camcol             | int64          | Camera column number, indicating which part of the imaging camera captured the object |
| field              | int64          | Field number in the run                                                               |
| specObjID          | float64        | Spectroscopic object identifier                                                       |
| class              | object         | Classification of the object (e.g., STAR, GALAXY)                                     |
| redshift           | float64        | Redshift of the object                                                                |
| plate              | int64          | Spectroscopic plate number                                                            |
| mjd                | int64          | Modified Julian Date of the observation                                               |
| fiberID            | int64          | Fiber ID for the spectroscopic observation                                            |

Below are the first few rows from the dataset:

| objID            | ra        | dec       | u       | g       | r       | i       | z       | run  | rerun | camcol | field | specObjID        | class  | redshift  | plate | mjd    | fiberID |
|------------------|-----------|-----------|---------|---------|---------|---------|---------|------|-------|--------|-------|------------------|--------|-----------|-------|--------|---------|
| 1.237650e+18     | 183.531326| 0.089693  | 19.47406| 17.04240| 15.94699| 15.50342| 15.22531| 752  | 301   | 4      | 267   | 3.722360e+18     | STAR   | -0.000009 | 3306  | 54922  | 491     |
| 1.237650e+18     | 183.598370| 0.135285  | 18.66280| 17.21449| 16.67637| 16.48922| 16.39150| 752  | 301   | 4      | 267   | 3.638140e+17     | STAR   | -0.000055 | 323   | 51615  | 541     |
| 1.237650e+18     | 183.680207| 0.126185  | 19.38298| 18.19169| 17.47428| 17.08732| 16.80125| 752  | 301   | 4      | 268   | 3.232740e+17     | GALAXY | 0.123111  | 287   | 52023  | 513     |
| 1.237650e+18     | 183.870529| 0.049911  | 17.76536| 16.60272| 16.16116| 15.98233| 15.90438| 752  | 301   | 4      | 269   | 3.722370e+18     | STAR   | -0.000111 | 3306  | 54922  | 510     |
| 1.237650e+18     | 183.883288| 0.102557  | 17.55025| 16.26342| 16.43869| 16.55492| 16.61326| 752  | 301   | 4      | 269   | 3.722370e+18     | STAR   | 0.000590  | 3306  | 54922  | 512     |

This dataset was chosen due to the simplicity of the classification problem it provides. There are only three classes (Star, Galaxy or QSO) so imbalances between them won't make a huge difference to the accuracy of the model, and most of the parameters are not relevent to the problem. The class column will become the labels (i.e. what we use to test the model). We can cut down the remaining 17 features to 6 by ignoring all of the ones that describe the telescope or metadata from the observations. This is because they shouldn't have much impact on classifying the objects unless they happen to contain patterns due to errors or biases in the observations. Either way we want to exclude these so the model doesn't learn false patterns.

This leaves us with the 5 photometric  magnitudes (u, g, r, i, and z), the Right Ascension (ra) and Declination (dec), and the redshift. For this problem, it is unlikely that the location of the object in the sky will be relevant so both ra and dec are removed. Redshift is also removed, not because it isn't relevant but because for this task it turned out to completely dominate the models.

For these reasons, I chose to only use the 5 photometric magnitudes to train both the DT and the NN.

# Required Packages
In order to run the code list in the notebooks, certain packages will need to be installed. These packages are listed in the [dependencies.txt](https://github.com/Ryan-Ursell/SDSS-DR14-Classification/blob/main/dependencies.txt) file.

These can be install all at once fairly easily by cloning the repository and then using pip to install from dependencies.txt:
```
git clone https://github.com/Ryan-Ursell/SDSS-DR14-Classification.git
cd SDSS-DR14-Classification
pip install -r dependencies.txt
```

**Important Note**: This notebook was made using Python version 3.13.1. PyTorch has not been officially updated to this version yet (as of January 2025) so I installed a develepment version form PyTorch's website [here](https://pytorch.org/get-started/locally/). The version is listed in dependencies.txt but it may not install automatically with pip due to this. If you wish to install the same version I used, you may have to grab it from PyTorch's website.

```
Dependencies:
- numpy - v2.2.1
- pandas - v2.2.3
- seaborn - v0.13.2
- matplotlib - v3.10.0
- scikit-learn - v1.6.0
- torch - v2.7.0.dev20250124+cpu
```

# Licence
This project is under the GNU General Public License v3.0, see [LICENCE](https://github.com/Ryan-Ursell/SDSS-DR14-Classification/blob/main/LICENSE) for more info.

# Acknowledgment
ChatGPT:
- Provider: OpenAI
- Description: ChatGPT was used to assist in generating and debugging portions of the codebase for this project

GitHub Copilot:
- Provider: GitHub & OpenAI
- ChatGPT was used to assist in generating and debugging portions of the codebase for this project

# Personal Information
Ryan Ursell

ryanursell@outlook.com

Mphys Physics, Astrophysics, and Cosmology

University of Portsmouth