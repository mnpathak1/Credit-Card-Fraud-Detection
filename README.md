# Credit Card Fraud Detection Analysis and Predction Model
## Dataset
This  dataset is taken from Kaggle:  
https://www.kaggle.com/mlg-ulb/creditcardfraud/data

We have time of transaction, 28 anonimyzed features, amount of transaction and the class of transaction in the dataset.
This datset is highly skewed with Fraud transactions being only 0.17% of total transactions.  

## Conclusion
* EDA did not show clear separation of fraud and normal transaction captured by any single parameter
* Fraud transactions are typically small. On a crude term, transaction with values more than maximum of fraud transaction can safely be assumed as normal
*  Both Tensorflow and Keras model built on the creditcard dataset showed very high test accuracies (99.46% & 99.82%) however, failed to capture the Fraud transaction in this highly skewed data
*  Autoencoder NN model with a small threshold for reconstruction error can capture most of Fraud transaction however, it also significantly misclassify Normal transaction as Fraud.
* t-SNE plot showed good separation between the normal and fraudalant transaction in the scatterplot suggesting prediction model to show good accuracy in model developed training and testing within the dataset.
## How to improve fraud detection?
* Undersampling of normal class data to match fraud sample size
* Otherway round, simulate (with SMOTE technique) more fraud data
* More fraud data is always better, particularly for NN
* Train a larger or different Autoencoder or other NN
