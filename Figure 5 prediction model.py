import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

file_path = "Figure 5 raw data.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

for subject_name, df in sheets.items():
    df = df.rename(columns={"Label": "Set"}) if "Label" in df.columns else df.copy()
    df = df.sort_values(by=["Detail Situation", "Set", "Minute Elapsed"])

    df["Base_UA"] = df.groupby(["Detail Situation", "Set"])["Uric Acid (μM)"].transform("first")
    df["ΔUA"] = df["Uric Acid (μM)"] - df["Base_UA"]

    X = df[["Situation", "Detail Situation", "Minute Elapsed", "Set"]]
    y = df["ΔUA"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Situation", "Detail Situation"]),
        ("num", StandardScaler(), ["Minute Elapsed", "Set"])
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
    ])

    # 학습
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # 저장
    file_name = f"dua_xgb_model_{subject_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, file_name)
    print(f"✅ Saved model: {file_name}")
