## Challenge + Dataset Source
ISIC 2024 Challenge - Skin Cancer Detection

## Inspiration
We currently have some experience in machine learning, statistics, mathematics, and data analysis, and we wanted a project to help us improve these skills. We were interested in working on an applicable project that generates a real-world impact. This project provided us the opportunity to delve into analyzing image data and image processing, something we always wanted to try out.

## What it does
We trained different models to predict whether growths were benign or malignant.

## How we built it
We used Jupyter notebooks in Python, matplotlib to visualize, and scikit-learn with other libraries to train our models.

## Challenges we ran into
* We didn't consider that the dataset would be so highly unbalanced. There were about 400 data points that had malignant growths compared to the roughly 400,000 data points in total.
* Feature selection was also difficult since there were about 50 features to choose from.
* Model training also proved to be slow on cloud computing.
* Working on the same notebook was also difficult since we wanted to test different blocks at the same time.
* We did not have the processing power to develop complex models.
* The dataset was much larger than any of us have worked with.

## Accomplishments that we're proud of
* We managed to overcome the challenge of an imbalanced dataset by learning and using the SMOTE-Tomek Links method.
* We explored different models and fine tuned one that exceeded our expectations.

## What we learned
* Ways to deal with imbalanced data (SMOTE-Tomek Links method)

## What's next for We Care for Skin Care
The next steps would be to continue exploring models that would provide better true positive rates and less false positives at the same time. Additionally, training models on only the camera images rather than full body scans would be helpful in the production of an app that people could use at home to see if they should go to the doctor.
