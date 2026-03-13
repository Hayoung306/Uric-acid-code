import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight


try:
    from sklearn.model_selection import StratifiedGroupKFold
    USE_STRATIFIED_GROUP = True
except:
    from sklearn.model_selection import GroupKFold
    USE_STRATIFIED_GROUP = False

warnings.filterwarnings("ignore")


file_path = 'Figure 4.xlsx'
data = pd.read_excel(file_path)

data = data[['subject_id','label','tear_difference','tear_rate','tear_AUC',
             'lag_time','blood_uric_acid','tear_uric_acid']].dropna()

feature_cols = ['tear_difference','tear_rate','tear_AUC',
                'lag_time','blood_uric_acid','tear_uric_acid']

X = data[feature_cols].reset_index(drop=True)
y = data['label'].reset_index(drop=True)
groups = data['subject_id'].reset_index(drop=True)

label_names = ['Healthy','Gout w/o treatment','Gout w/ treatment']
all_labels = sorted(y.unique())
n_classes = len(all_labels)


outer_splits = 3
inner_splits = 2
random_state = 42

if USE_STRATIFIED_GROUP:
    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_iterator = outer_cv.split(X,y,groups)
else:
    outer_cv = GroupKFold(n_splits=outer_splits)
    outer_iterator = outer_cv.split(X,y,groups)

param_grid = {
    'max_depth':[2,3],
    'learning_rate':[0.03,0.05,0.1],
    'n_estimators':[30,50,80],
    'subsample':[0.7,0.9],
    'colsample_bytree':[0.7,0.9],
    'reg_alpha':[0,1],
    'reg_lambda':[1,5]
}

all_y_true=[]
all_y_pred=[]
feature_importance_list=[]
fold_results=[]


for fold,(train_idx,test_idx) in enumerate(outer_iterator,1):

    X_train,X_test = X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test = y.iloc[train_idx],y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    best_score=-np.inf
    best_params=None

    if USE_STRATIFIED_GROUP:
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits,shuffle=True,random_state=random_state)
        inner_iterator = inner_cv.split(X_train,y_train,groups_train)
    else:
        inner_cv = GroupKFold(n_splits=inner_splits)
        inner_iterator = inner_cv.split(X_train,y_train,groups_train)

    for params in ParameterGrid(param_grid):

        scores=[]

        for inner_train,inner_val in inner_iterator:

            X_it = X_train.iloc[inner_train]
            X_val = X_train.iloc[inner_val]
            y_it = y_train.iloc[inner_train]
            y_val = y_train.iloc[inner_val]

            if y_it.nunique()<2:
                continue

            weights = compute_sample_weight(class_weight='balanced',y=y_it)

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                random_state=random_state,
                **params
            )

            model.fit(X_it,y_it,sample_weight=weights)

            y_val_pred = model.predict(X_val)

            score = accuracy_score(y_val,y_val_pred)
            scores.append(score)

        if len(scores)==0:
            continue

        mean_score = np.mean(scores)

        if mean_score>best_score:
            best_score=mean_score
            best_params=params

    if best_params is None:
        best_params = {
            'max_depth':2,
            'learning_rate':0.05,
            'n_estimators':50,
            'subsample':0.8,
            'colsample_bytree':0.8,
            'reg_alpha':0,
            'reg_lambda':1
        }

    weights = compute_sample_weight(class_weight='balanced',y=y_train)

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        random_state=random_state,
        **best_params
    )

    final_model.fit(X_train,y_train,sample_weight=weights)

    y_pred = final_model.predict(X_test)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    feature_importance_list.append(final_model.feature_importances_)


cm = confusion_matrix(all_y_true,all_y_pred,labels=all_labels)

cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]*100

print("Confusion Matrix (%):")
print(cm_pct)

print("\nClassification Report")
print(classification_report(all_y_true,all_y_pred,target_names=label_names))

plt.figure(figsize=(8,6))
sns.heatmap(cm_pct,annot=True,fmt='.2f',cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names)
plt.title("Confusion Matrix (Nested Group CV)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

importance_array = np.array(feature_importance_list)

importance_mean = importance_array.mean(axis=0)

importance_df = pd.DataFrame({
    'Feature':feature_cols,
    'Importance':importance_mean
}).sort_values(by='Importance',ascending=False)

importance_df['Normalized Importance'] = importance_df['Importance']/importance_df['Importance'].sum()

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df,
            x='Normalized Importance',
            y='Feature',
            color='skyblue')

plt.title("Feature Importance (XGBoost)",fontsize=18)
plt.xlabel("Normalized Importance")
plt.ylabel("Feature")

for i,v in enumerate(importance_df['Normalized Importance']):
    plt.text(v+0.01,i,f"{v:.2f}",va='center')

plt.tight_layout()
plt.show()
