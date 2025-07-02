#  Car Price Prediction

This repository contains a Jupyter Notebook (`CarPricePrediction .ipynb`) that focuses on predicting car prices based on a variety of features. The project involves comprehensive data cleaning, preprocessing, feature engineering, and training several machine learning models.

## Table of Contents

  - [Project Overview](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#project-overview)
  - [Dataset](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#dataset)
  - [Features](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#features)
  - [Preprocessing and Feature Engineering](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#preprocessing-and-feature-engineering)
  - [Exploratory Data Analysis (EDA)](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#exploratory-data-analysis-eda)
  - [Model Training](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#model-training)
  - [Model Performance](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#model-performance)
  - [Prediction Example](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#prediction-example)
  - [Getting Started](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#getting-started)
  - [Libraries Used](https://github.com/nouninasion/CarPricePrediction/blob/main/README.md#libraries-used)

## Project Overview

The primary goal of this project is to develop a machine learning model capable of accurately predicting car prices. The notebook demonstrates a typical data science workflow, from initial data loading and exploration to advanced feature engineering and model evaluation using regression algorithms.

## Dataset

The dataset used for this prediction task is `car_price_prediction.csv`. It includes various attributes of cars that are crucial for price estimation.

## Features

The dataset contains the following features:

  - `Levy`: A fee or tax associated with the car.
  - `Manufacturer`: The brand of the car.
  - `Model`: The specific model of the car.
  - `Prod. year`: The production year of the car.
  - `Category`: The type of car (e.g., Sedan, SUV).
  - `Leather interior`: Indicates if the car has a leather interior.
  - `Fuel type`: The type of fuel the car uses.
  - `Engine volume`: The engine displacement.
  - `Mileage`: The mileage of the car.
  - `Cylinders`: The number of cylinders in the engine.
  - `Gear box type`: The type of transmission.
  - `Drive wheels`: The drive wheel configuration.
  - `Doors`: The number of doors.
  - `Wheel`: The wheel steering side (e.g., Left, Right).
  - `Airbags`: The number of airbags.
  - `Price`: The price of the car (target variable).

## Preprocessing and Feature Engineering

Extensive preprocessing and feature engineering steps were performed to prepare the data for modeling:

  - **Handling Missing Values**: Missing values in object (categorical) columns were filled with the mode, while numerical columns had their missing values imputed with the mean.
  - **Duplicate Removal**: Duplicate rows in the dataset were identified and removed.
  - **Feature Cleaning and Transformation**:
      - `Price`, `Levy`, and `Mileage` columns were cleaned by removing non-numeric characters (like '$', ',', 'km') and converted to numeric types.
      - A log transformation (`np.log1p`) was applied to `Price`, `Levy`, `Mileage`, and `Engine volume` to handle their skewed distributions.
  - **Categorical Encoding**:
      - `Manufacturer`, `Model`, `Category`, `Fuel type`, `Gear box type`, `Drive wheels`, `Doors`, and `Wheel` columns were encoded using `LabelEncoder`.
      - The `Engine volume` column, which sometimes contained "Turbo" or "Hybrid", was processed to extract the numerical volume and a new binary feature `Is_Turbo` was created. "Hybrid" values were mapped to a numerical representation based on an arbitrary average.
  - **Feature Scaling**: All features were scaled using `StandardScaler` to normalize their range.

## Exploratory Data Analysis (EDA)

The notebook includes initial data inspection using `data.info()` and `data.describe()` to understand data types, non-null counts, and basic statistics. Histograms for numerical features were plotted to visualize their distributions, and a heatmap of the correlation matrix was generated to identify relationships between variables.

## Model Training

The processed data was split into training and testing sets. Two regression models were trained:

1.  **Linear Regression**: A simple linear model to serve as a baseline.
2.  **RandomForestRegressor**: An ensemble learning method, typically robust and accurate for complex datasets.

## Model Performance

The performance of the models was evaluated using the R-squared score on the test set:

  - **Linear Regression**: R-squared score: `0.584`.
  - **RandomForestRegressor**: R-squared score: `0.954`.

The RandomForestRegressor significantly outperformed the Linear Regression model, indicating its effectiveness in capturing the intricate relationships within the car price dataset.

## Prediction Example

The notebook provides an example of how to use the trained RandomForestRegressor model to predict the price of a new, hypothetical car.

## Getting Started

To run this notebook, you will need Google Colab or a local Jupyter environment with the following libraries installed:

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tabulate` (for displaying prediction results nicely)

You can install these using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```

Once the dependencies are installed, you can open and run the `CarPricePrediction .ipynb` file in your Google Colab or Jupyter environment.

## Libraries Used

  - `pandas`
  - `numpy`
  - `matplotlib.pyplot`
  - `seaborn`
  - `sklearn.model_selection`
  - `sklearn.impute`
  - `sklearn.preprocessing`
  - `sklearn.linear_model`
  - `sklearn.ensemble`
  - `sklearn.tree`
  - `re` (Python's built-in regular expression module)
  - `tabulate`
