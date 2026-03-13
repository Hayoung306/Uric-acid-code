import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from sklearn.model_selection import StratifiedGroupKFold
    USE_STRATIFIED_GROUP = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    USE_STRATIFIED_GROUP = False

warnings.filterwarnings("ignore")


file_path = 'Figure 4.xlsx'
data = pd.read_excel(file_path)

data = data[['subject_id', 'label', 'blood_uric_acid', 'tear_uric_acid']].dropna().reset_index(drop=True)

feature_cols = ['blood_uric_acid', 'tear_uric_acid']
X = data[feature_cols].reset_index(drop=True)
y = data['label'].reset_index(drop=True)
groups = data['subject_id'].reset_index(drop=True)

all_labels = sorted(y.unique())
label_name_map = {
    0: 'Healthy',
    1: 'Gout w/o treatment',
    2: 'Gout w/ treatment'
}
label_names = [label_name_map[l] for l in all_labels]
n_classes = len(all_labels)

print("Unique labels:", all_labels)
print("\nSample counts:")
print(y.value_counts().sort_index())
print("\nSubject counts by class:")
print(data.groupby('label')['subject_id'].nunique())


def safe_balanced_accuracy_multiclass(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1)
    recalls = []

    for i in range(len(labels)):
        if row_sums[i] > 0:
            recalls.append(cm[i, i] / row_sums[i])

    if len(recalls) == 0:
        return 0.0
    return float(np.mean(recalls))


outer_n_splits = 3
inner_n_splits = 2
random_state = 42

if USE_STRATIFIED_GROUP:
    outer_cv = StratifiedGroupKFold(
        n_splits=outer_n_splits,
        shuffle=True,
        random_state=random_state
    )
    outer_splitter = outer_cv.split(X, y, groups)
else:
    outer_cv = GroupKFold(n_splits=outer_n_splits)
    outer_splitter = outer_cv.split(X, y, groups)

param_grid = {
    'max_depth': [2, 3],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [30, 50, 80],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'reg_alpha': [0, 1],
    'reg_lambda': [1, 5]
}


all_y_true = []
all_y_pred = []
fold_results = []
feature_importance_list = []


for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter, start=1):
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    print(f"\n========== Outer Fold {fold_idx} ==========")
    print("Train counts:")
    print(y_train.value_counts().sort_index())
    print("Test counts:")
    print(y_test.value_counts().sort_index())

    best_score = -np.inf
    best_params = None

    if groups_train.nunique() >= inner_n_splits:
        if USE_STRATIFIED_GROUP:
            inner_cv = StratifiedGroupKFold(
                n_splits=inner_n_splits,
                shuffle=True,
                random_state=random_state
            )
        else:
            inner_cv = GroupKFold(n_splits=inner_n_splits)

        for params in ParameterGrid(param_grid):
            inner_scores = []

            if USE_STRATIFIED_GROUP:
                current_splitter = inner_cv.split(X_train, y_train, groups_train)
            else:
                current_splitter = inner_cv.split(X_train, y_train, groups_train)

            for inner_train_idx, inner_val_idx in current_splitter:
                X_inner_train = X_train.iloc[inner_train_idx].reset_index(drop=True)
                X_inner_val = X_train.iloc[inner_val_idx].reset_index(drop=True)
                y_inner_train = y_train.iloc[inner_train_idx].reset_index(drop=True)
                y_inner_val = y_train.iloc[inner_val_idx].reset_index(drop=True)


                if y_inner_train.nunique() < 2:
                    continue

                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=y_inner_train
                )

                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    eval_metric='mlogloss',
                    random_state=random_state,
                    use_label_encoder=False,
                    **params
                )

                model.fit(
                    X_inner_train,
                    y_inner_train,
                    sample_weight=sample_weights
                )

                y_val_pred = model.predict(X_inner_val)

                score = safe_balanced_accuracy_multiclass(
                    y_true=y_inner_val,
                    y_pred=y_val_pred,
                    labels=all_labels
                )
                inner_scores.append(score)

            if len(inner_scores) == 0:
                continue

            mean_score = np.mean(inner_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params


    if best_params is None:
        best_params = {
            'max_depth': 2,
            'learning_rate': 0.05,
            'n_estimators': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1
        }

    print("Best params:", best_params)

    outer_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        random_state=random_state,
        use_label_encoder=False,
        **best_params
    )

    final_model.fit(
        X_train,
        y_train,
        sample_weight=outer_weights
    )

    y_test_pred = final_model.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = safe_balanced_accuracy_multiclass(y_test, y_test_pred, all_labels)
    f1_macro = f1_score(
        y_test,
        y_test_pred,
        average='macro',
        labels=all_labels,
        zero_division=0
    )

    fold_results.append({
        'fold': fold_idx,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'best_params': best_params
    })

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_test_pred.tolist())

    feature_importance_list.append(final_model.feature_importances_)


results_df = pd.DataFrame(fold_results)

print("\n==============================")
print("Per-fold results")
print("==============================")
print(results_df[['fold', 'accuracy', 'balanced_accuracy', 'f1_macro']])

print("\nMean ± SD:")
for metric in ['accuracy', 'balanced_accuracy', 'f1_macro']:
    print(f"{metric}: {results_df[metric].mean():.3f} ± {results_df[metric].std():.3f}")

print("\n==============================")
print("Classification Report")
print("==============================")
print(classification_report(
    all_y_true,
    all_y_pred,
    labels=all_labels,
    target_names=label_names,
    digits=3,
    zero_division=0
))


cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)

row_sums = cm.sum(axis=1, keepdims=True)
cm_percentage = np.divide(
    cm.astype(float) * 100,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

print("\n==============================")
print("Confusion Matrix (row-normalized %)")
print("==============================")
print(cm_percentage)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_percentage,
    annot=True,
    fmt='.2f',
    cmap='Purples',
    xticklabels=label_names,
    yticklabels=label_names,
    square=True
)
plt.title('Confusion Matrix (Nested Grouped XGBoost)\n(Blood + Tear Uric Acid)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.show()


feature_importance_array = np.array(feature_importance_list)

importance_mean = feature_importance_array.mean(axis=0)
importance_std = feature_importance_array.std(axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance Mean': importance_mean,
    'Importance Std': importance_std
}).sort_values(by='Importance Mean', ascending=False).reset_index(drop=True)

total_importance = importance_df['Importance Mean'].sum()
if total_importance > 0:
    importance_df['Normalized Importance'] = importance_df['Importance Mean'] / total_importance
    importance_df['Normalized Std'] = importance_df['Importance Std'] / total_importance
else:
    importance_df['Normalized Importance'] = 0
    importance_df['Normalized Std'] = 0

print("\n==============================")
print("Feature importance across outer folds")
print("==============================")
print(importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=importance_df,
    x='Normalized Importance',
    y='Feature',
    color='purple'
)

plt.title('Feature Importance (Blood + Tear Uric Acid)', fontsize=16)
plt.xlabel('Normalized Mean Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)

for i, row in importance_df.iterrows():
    plt.text(
        row['Normalized Importance'] + 0.01,
        i,
        f"{row['Normalized Importance']:.3f}",
        va='center',
        fontsize=12
    )

plt.tight_layout()
plt.show()