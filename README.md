## Challenge + Dataset Source
ISIC 2024 Challenge - Skin Cancer Detection

## Inspiration
We currently have some experience in machine learning, statistics, mathematics, and data analysis, and we wanted a project to help us improve these skills. We were interested in working on an interesting and applicable project that generates a real-world impact. This project provided us the opportunity to delve into analyzing image data and image processing, something we always wanted to try out.

## What it does
Our project analyzes both 2D and 3D image data of skin lesions and predicts whether a lesion is malignant or not using a trained AI model. We wanted to build something meaningful, and this gave us a chance to explore computer vision while working on a real-world health-related problem.

## How we built it
We started by researching melanoma and skin cancer to better understand the medical context behind our dataset. After cleaning the data by removing null columns, dropping training-only features, and converting categorical variables into numerical ones, we used MatPlotLib to visualize key features and identify patterns related to malignancy. We trained a logistic regression model and a convolutional neural network (CNN) using PyTorch to classify the skin lesions. To address the unbalanced dataset, we applied the SMOTE-Tomek Links method, which significantly improved our model’s accuracy—from just 1% true positives to nearly 70%. Along the way, we also experimented with different feature selection strategies and evaluated our models using confusion matrices and ROC curves.

## Challenges we ran into
* We didn't consider that the dataset would be so highly unbalanced. There were about 400 data points that had malignant growths compared to the roughly 400,000 data points in total.
* Feature selection was also difficult since there were about 50 features to choose from.
* Model training also proved to be slow on cloud computing.
* Working on the same notebook was also difficult since we wanted to test different blocks at the same time.

## Accomplishments that we're proud of
* We managed to overcome the challenge of an unbalanced dataset by learning and using the SMOTE-Tomek Links method.
* We developed a model that performed relatively well.

## What we learned
* SMOTE-Tomek Links method

## What's next for We Care for Skin Care

