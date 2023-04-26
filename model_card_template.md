# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- `Random Forest` is an ensemble method that combines multiple decision trees to improve the accuracy and stability of predictions. It is a popular machine learning algorithm that can be used for both classification and regression tasks.

- In `scikit-learn`, the `RandomForestClassifier` is implemented using the CART (Classification and Regression Trees) algorithm, which is a recursive partitioning method that splits the data into subsets based on the most discriminatory features. The algorithm continues to recursively partition the subsets until a stopping criterion is met, such as a maximum tree depth or minimum number of samples required to split a node.

## Intended Use
The intended use of the `Random Forest model` is to predict the salary of a person based on their financial attributes. This can be useful in a variety of settings, such as human resources and financial planning. For example, a company could use this model to predict the salary of a job candidate based on their financial history, which could help inform the negotiation process. Similarly, an individual could use this model to estimate their future earnings potential based on their financial situation. However, it is important to note that any use of this model should be done with caution and in compliance with relevant laws and ethical considerations. Additionally, the model should be evaluated carefully to ensure that it is accurate and fair, and that it does not unfairly discriminate against any particular group.

## Training Data
Using `80%` of this data: `https://archive.ics.uci.edu/ml/datasets/census+income` for training model.

## Evaluation Data
Using `20%` of this data: `https://archive.ics.uci.edu/ml/datasets/census+income` for evaluating model

## Metrics
- Accuracy score: ~ `0.81`.

## Ethical Considerations
The dataset includes information about race, gender, and country of origin, which could potentially result in discriminatory predictions. As such, it is crucial to conduct a thorough investigation before using the model to ensure that it does not unfairly discriminate against any particular group.

## Caveats and Recommendations
The gender classes in the dataset are limited to a binary classification of "male" or "not male", which we have included as "male" and "female" respectively. However, further analysis is required to evaluate the model's performance across a wider spectrum of gender identities.