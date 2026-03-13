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
except ImportError:
    from sklearn.model_selection import GroupKFold
    USE_STRATIFIED_GROUP = False

warnings.filterwarnings("ignore")


file_path = 'Figure 4.xlsx'
data = pd.read_excel(file_path)

data = data[['subject_id', 'label', 'tear_uric_acid', 'blood_uric_acid']].dropna().reset_index(drop=True)
data = data[data['label'] != 2].reset_index(drop=True)

label_names = ['Healthy', 'Gout w/o treatment']
all_labels = [0, 1]

print("Unique labels:", sorted(data['label'].unique()))
print("\nSample counts:")
print(data['label'].value_counts().sort_index())
print("\nSubject counts by class:")
print(data.groupby('label')['subject_id'].nunique())



def safe_balanced_accuracy_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1)
    recalls = []
    for i in range(2):
        if row_sums[i] > 0:
            recalls.append(cm[i, i] / row_sums[i])
    if len(recalls) == 0:
        return 0.0
    return float(np.mean(recalls))


def run_nested_grouped_xgb_binary(
    data,
    feature_cols,
    outer_n_splits=3,
    inner_n_splits=2,
    random_state=42
):
    X = data[feature_cols].reset_index(drop=True)
    y = data['label'].reset_index(drop=True)
    groups = data['subject_id'].reset_index(drop=True)

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


                    if y_inner_train.nunique() < 2 or y_inner_val.nunique() < 2:
                        continue

                    sample_weights = compute_sample_weight(
                        class_weight='balanced',
                        y=y_inner_train
                    )

                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        eval_metric='logloss',
                        random_state=random_state,
                        use_label_encoder=False,
                        **params
                    )

                    model.fit(
                        X_inner_train,
                        y_inner_train,
                        sample_weight=sample_weights
                    )

                    y_val_prob = model.predict_proba(X_inner_val)[:, 1]
                    y_val_pred = (y_val_prob >= 0.5).astype(int)

                    score = safe_balanced_accuracy_binary(y_inner_val, y_val_pred)
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

        outer_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False,
            **best_params
        )

        final_model.fit(
            X_train,
            y_train,
            sample_weight=outer_weights
        )

        y_test_prob = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_test_pred)
        bal_acc = safe_balanced_accuracy_binary(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)

        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1': f1,
            'best_params': best_params
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_test_pred.tolist())
        feature_importance_list.append(final_model.feature_importances_)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm.astype(float) * 100,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    results_df = pd.DataFrame(fold_results)

    importance_array = np.array(feature_importance_list)
    importance_mean = importance_array.mean(axis=0)
    importance_std = importance_array.std(axis=0)

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

    return {
        'results_df': results_df,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'cm': cm,
        'cm_pct': cm_pct,
        'importance_df': importance_df
    }


feature_sets = {
    'Tear Only': ['tear_uric_acid'],
    'Blood Only': ['blood_uric_acid'],
    'Tear + Blood': ['tear_uric_acid', 'blood_uric_acid']
}

cm_colors = {
    'Tear Only': 'Blues',
    'Blood Only': 'Reds',
    'Tear + Blood': 'Purples'
}

results = {}

for model_name, feature_cols in feature_sets.items():
    print(f"\n==============================")
    print(f"Model: {model_name}")
    print(f"Features: {feature_cols}")
    print(f"==============================")

    out = run_nested_grouped_xgb_binary(
        data=data,
        feature_cols=feature_cols,
        outer_n_splits=3,
        inner_n_splits=2,
        random_state=42
    )
    results[model_name] = out

    print("\nPer-fold results:")
    print(out['results_df'][['fold', 'accuracy', 'balanced_accuracy', 'f1']])

    print("\nMean ± SD:")
    for metric in ['accuracy', 'balanced_accuracy', 'f1']:
        print(f"{metric}: {out['results_df'][metric].mean():.3f} ± {out['results_df'][metric].std():.3f}")

    print("\nClassification Report:")
    print(classification_report(
        out['all_y_true'],
        out['all_y_pred'],
        labels=all_labels,
        target_names=label_names,
        digits=3,
        zero_division=0
    ))

    print("\nConfusion Matrix (%):")
    print(out['cm_pct'])


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, model_name in enumerate(['Tear Only', 'Blood Only', 'Tear + Blood']):
    sns.heatmap(
        results[model_name]['cm_pct'],
        annot=True,
        fmt='.2f',
        cmap=cm_colors[model_name],
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[idx],
        cbar=False,
        square=True,
        annot_kws={'size': 14}
    )
    axes[idx].set_title(model_name, fontsize=16)
    axes[idx].set_xlabel('Predicted', fontsize=14)
    if idx == 0:
        axes[idx].set_ylabel('Actual', fontsize=14)
    else:
        axes[idx].set_ylabel('')

plt.tight_layout()
plt.show()


importance_df = results['Tear + Blood']['importance_df']

plt.figure(figsize=(8, 5))
sns.barplot(
    data=importance_df,
    x='Normalized Importance',
    y='Feature',
    color='purple'
)

plt.title('Feature Importance (Tear + Blood)', fontsize=18)
plt.xlabel('Normalized Mean Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)

for i, row in importance_df.iterrows():
    plt.text(
        row['Normalized Importance'] + 0.01,
        i,
        f"{row['Normalized Importance']:.2f}",
        va='center',
        fontsize=12
    )

plt.tight_layout()
plt.show()