# Glossary

Use this page to record terms and ideas that help you understand
professional analytics projects.

This project covers feature engineering and selection:
preparing raw data so a machine learning model can learn from it effectively.

Pro-tip: Expand the VS Code **Outline** view (below the navigator on the right)
to see this file organization at-a-glance.

## Data and Outputs

### raw data

Raw data is the original input data used by the project.
It should usually be kept unchanged so the analysis
can be repeated from the original source.

### processed data

Processed data is data that has been
cleaned, filtered, transformed, summarized, or prepared for later use.

### artifact

An artifact is a file created by running a project.
Examples include logs, charts, reports, and exported data files.

## Feature Engineering

### feature engineering

Feature engineering is the process of creating, transforming, or selecting
input columns to improve model performance.
Good features often matter more than the choice of model.

### missing values

Missing values are cells in a dataset with no recorded entry.
Common strategies include removing rows with missing values,
filling them with the mean or median, or using a placeholder category.

### imputation

Imputation is the process of filling in missing values with estimated ones.
The choice of imputation strategy is an analyst decision.

### encoding

Encoding is the process of converting categorical text values into numbers
so a model can work with them.
One-hot encoding creates a separate binary column for each category.
Ordinal encoding assigns an integer to each category.

### one-hot encoding

One-hot encoding converts a categorical column into multiple binary columns,
one per category.
Each row has a 1 in the column for its category and 0 in all others.

### scaling

Scaling transforms numeric features so they fall within a similar range.
This prevents features with large values from dominating those with small values.

### standardization

Standardization rescales a feature so it has a mean of 0 and a standard deviation of 1.
It is a common scaling method when features have different units or ranges.

### normalization

Normalization rescales a feature so all values fall between 0 and 1.
It is useful when the model is sensitive to absolute value ranges.

### feature selection

Feature selection is the process of choosing which input columns to keep
and which to remove before training a model.
Removing irrelevant or redundant features can improve accuracy and reduce complexity.

### correlation

Correlation measures how closely two variables move together.
High correlation between a feature and the target suggests the feature
may be useful for prediction.
High correlation between two features suggests one may be redundant.

### outlier

An outlier is a data point that is very different from the rest.
Outliers can distort model training and may need to be removed or capped.
