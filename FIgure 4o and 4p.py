import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'Figure 4 raw data.xlsx'


data = pd.read_excel(file_path)


data = data[['subject_id', 'label', 'tear_difference', 'tear_rate', 'tear_AUC', 'lag_time', 'blood_uric_acid', 'tear_uric_acid']]
data = data.dropna()


features = data[['tear_difference', 'tear_rate', 'tear_AUC', 'lag_time', 'blood_uric_acid', 'tear_uric_acid']]
labels = data['label']


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)


params = {
    'objective': 'multi:softmax',
    'num_class': len(labels.unique()),
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}


xgboost_model = xgb.train(params, train_data, num_boost_round=100)


y_pred = xgboost_model.predict(test_data)


label_names = ['Healthy', 'Gout w/o treatment', 'Gout w/ treatment']


cm = confusion_matrix(y_test, y_pred)


cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100


print("Confusion Matrix (행 기준 백분율):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_names))


plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix (XGBoost) - Percentage per Class')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
