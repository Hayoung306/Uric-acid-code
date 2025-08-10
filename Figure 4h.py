import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'Figure 4 raw data'


data = pd.read_excel(file_path)


data = data[['label', 'tear_uric_acid']].dropna()  #


data = data[data['label'] != 0]
data['label'] = data['label'].replace({1: 0, 2: 1})  #


features = data[['tear_uric_acid']]
labels = data['label']


scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=['tear_uric_acid'])


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 5,  #
    'colsample_bytree': 0.8  #
}


xgboost_model = xgb.train(params, train_data, num_boost_round=100)


y_pred = xgboost_model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)


label_names = ['Gout w/o treatment', 'Gout w/ treatment']
cm = confusion_matrix(y_test, y_pred_binary)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("Confusion Matrix (Percentage by Row):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary, target_names=label_names))


plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'}, annot_kws={"size": 14})
plt.title('Confusion Matrix (XGBoost) - Tear Uric Acid Classification', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('Actual Label', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


plt.figure(figsize=(8, 6))
sns.barplot(x='Normalized Importance', y='Feature', data=importance_df, color='blue')
plt.title('Feature Importance (XGBoost)', fontsize=16)
plt.xlabel('Normalized Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
