# Model Card

## Model Details
 - Developed by Heather Bassler, student at Western Governors University
 - Model Date: 2026-04-19
 - Model Version: 1.0.0
 - Model Type: Logistic Regression

## Intended Use
 - This model is intended to estimate whether individal salary is above or below $50k from US Census data.
 - This model is not intended to estimate household level income.
 - This model is not intended to predict actual income of an individual.

## Training Data
 - The model is trained using a training subset of census.csv.

## Evaluation Data
 - The model was evaluated using a testing subset of census.csv.

## Metrics
The overall model has a precision of 0.7172, indicating that when it predicts an income over 50K, it is correct about 72% of the time. The recall of 0.2680 indicates that the model identifies only about 27% of all people with incomes over 50K, which suggests a high number of false negatives. The F1 score of 0.3902 reflects the balance between precision and recall, and its relatively low value indicates weak overall performance driven mainly by low recall.

The raw data was disproportionate, with about 76% of the records showing incomes under 50K. This class imbalance likely contributes to the model’s low recall, as the model appears biased toward predicting the majority class.

In reviewing individual feature slices, the model performed better for males (F1 = 0.4011) than for females (F1 = 0.3303), suggesting uneven performance across subgroups.

Performance also varies substantially across occupations. The model performs comparatively better for Sales, Exec‑managerial, Tech‑support, and Prof‑specialty roles (F1 scores between 0.4055 and 0.4345) than for most other occupations, many of which fall in the 0.0–0.3902 range. This again highlights inconsistent behavior across feature slices.

## Ethical Considerations
The training data for this model is highly skewed, resulting in class imbalance. This biases results towards the majority class.

## Caveats and Recommendations
Resampling census data is highly recommended to reduce bias.