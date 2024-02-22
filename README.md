# ML-Fairness-in-School-DropOut

This project aims to address the issue of fairness in machine learning models for predicting school dropout rates. The goal is to develop and evaluate models that are not biased towards any particular group, ensuring equal opportunities for all students.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The aim of this project is to utilize various machine learning models to predict the school dropout rate. Additionally, we evaluate the fairness of these models in reference with different definitions of fairness.

## Installation
To facilitate an easy setup, we have provided a .env file. You can use this file to install all the necessary packages required for this project by executing a single command.

## Data
This dataset originates from Kaggle, titled "Predict students' dropout and academic success: Investigating the Impact of Social and Economic Factors." The dataset is comprehensively described, including its source, format in the Kaggle website. For public datasets, you can [downlaod it here ](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data).

## Model Training
We trained 3 different ML models:
 - Logistic Regression
 - Random Forest
 - Decision Trees

## Evaluation
Our evaluation metrics include:

- **Accuracy:** The accuracy rate is a measure of how many instances were correctly classified by the model, where a higher accuracy rate generally indicates a better-performing model.

- **False Positive Rate:**  Calculates the false positive rate (FPR) for a binary classification task. The FPR is a measure of how many negative instances (i.e., instances that should have been classified as 0) were incorrectly classified as positive (1).

- **Recall Rate:** Calculates the recall rate (also known as sensitivity or true positive rate) for a binary classification task. The recall rate is a measure of how many positive instances (i.e., instances that should have been classified as 1) were correctly classified as positive by the model. It's a crucial metric in situations where the cost of missing a positive instance is high.

- **Demographic Parity:** Demographic parity is a fairness metric that measures whether the positive prediction rates are equal across different groups. In other words, it checks if individuals from different groups have an equal chance of being predicted as positive, regardless of their true labels.

- **Equality of Opportunity:** Equality of opportunity is a fairness metric that measures whether the true positive rates are equal across different groups. In other words, it checks if individuals who should have been classified as positive (i.e., true label is 1) have an equal chance of being correctly classified as positive, regardless of their group membership.



## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2024 School dropout rate

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact
If you have any questions or suggestions, feel free to contact [Deborah D Kanubala](mailto:dkanubala@aimsammi.org) or [Alidu Abubakari](mailto:alidu.abubakari@acity.edu.gh).

