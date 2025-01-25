# Classification of Celestial Objects from SDSS DR14
Classification of Stars, Galaxies, and Quasars (QSO) from the Sloan Digital Sky Servey (SDSS) Data Release 14 (DR14)

# Project Overview
The aim of this project is to compare two different classification methods, a traditional machine learning (ML) technique and a Neural Network (NN), using real world data from SDSS DR14. Both methods will attempt to classify Stars, Galaxies, and Quasars, something that is essential in Astronomy. Often trying to distinquish between these objects can be difficult as they are all similar "point-like" objects on the night sky, so training a ML algorithm to be able to identify them is incredibly useful.

The project is structured into three main notebooks. Question 1 focuses on explaining what a Decision Tree is and how it can be used to classify objects; Question 2 focuses on explaining how a Feed Forward Neural Network can be used; and Question 3 explores how changing certain parts of the NN in Question 2 can affect the model's performance.

# Dataset
The dataset used for this project comes from [Kaggle](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey/data).

To learn more about SDSS click [here](https://skyserver.sdss.org/dr14/en/home.aspx).

For SDSS's glossary click [here](https://live-sdss4org-dr14.pantheonsite.io/help/glossary).

The data contains 18 columns and 10,000 rows. Below are the columns and what they represent:

| Column Name        | Data Type      | Description                                                                            |
|--------------------|----------------|----------------------------------------------------------------------------------------|
| objID              | float64        | Unique identifier for each astronomical object in the catalog.                         |
| ra                 | float64        | Right Ascension (RA) of the object in decimal degrees.                                 |
| dec                | float64        | Declination (DEC) of the object in decimal degrees.                                    |
| u                  | float64        | Magnitude of the object in the SDSS U filter.                                          |
| g                  | float64        | Magnitude of the object in the SDSS G filter.                                          |
| r                  | float64        | Magnitude of the object in the SDSS R filter.                                          |
| i                  | float64        | Magnitude of the object in the SDSS I filter.                                          |
| z                  | float64        | Magnitude of the object in the SDSS Z filter.                                          |
| run                | int64          | Run number of the observation.                                                         |
| rerun              | int64          | Rerun number for calibration.                                                          |
| camcol             | int64          | Camera column number, indicating which part of the imaging camera captured the object. |
| field              | int64          | Field number in the run.                                                               |
| specObjID          | float64        | Spectroscopic object identifier.                                                       |
| class              | object         | Classification of the object (e.g., STAR, GALAXY).                                     |
| redshift           | float64        | Redshift of the object.                                                                |
| plate              | int64          | Spectroscopic plate number.                                                            |
| mjd                | int64          | Modified Julian Date of the observation.                                               |
| fiberID            | int64          | Fiber ID for the spectroscopic observation.                                            |
