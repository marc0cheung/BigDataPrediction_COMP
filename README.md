# BigDataPrediction_COMP
kNN (K-Nearest Neighbours) &amp; BP Neural Network solution in Python for data prediction. Sense the power of big data analytics to generate accurate results without expert knowledge of a certain application domain. 
<br></br>

## Data Description
For each dataset, we provide its training data (training.csv), validation data (validation.csv), and testing data (test.csv).
<br></br>

### Data 1
_**Training data:** training_1.csv_
- Each row is a data record containing 17 attributes
- The last cell of each row is class label in integer
- Training data includes training records with ground-truth class labels
- Use the training data to train your solution
- During training, do not let your solution access the following validation and testing datasets

_**Validation data:** validation_1.csv_
- After training the solutions, use this validation data to evaluate the performance of the solution (This validation data includes the ground-truth class labels)
- You can use third-party libraries to calculate these evaluation metrics
- A detailed explanation on the calculation of the metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
- **Example:** A third-party library to calculate the Micro-F1 and Macro-F1 scores: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

_**Testing data:** testing_1.csv_
- The `class label` column in testing.csv is empty
- Use your method to get predicted labels of the testing records in testing.csv, and input the predicted labels in the respective cells of the `class label` column in testing.csv
- Do not change the row order of the records in this file

<br></br>

### Data 2
_**Training data:** training_2.csv_
- Each row is a data record containing 15 attributes
- The last cell of each row is class label in integer
- Training data includes training records with ground-truth class labels
- Use the training data to train your solution
- During training, do not let your solution access the following validation and testing datasets

_**Validation data:** validation_2.csv and Testing data: testing_2.csv_
- Same instructions explained above in Data1 testing and validation data

<br></br>
