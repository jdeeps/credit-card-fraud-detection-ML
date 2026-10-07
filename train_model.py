"""
train_model.py
Trains the fraud detection model used by app.py and saves it to model.pkl.

Usage:
    python train_model.py

Expects a CSV at Data_Processing/FraudData_ValidationSet.csv (same file used
in 04.Supervised_Modelling.ipynb) with an 'is_fraud' label column and the
24 feature columns listed below. Adjust DATA_PATH if yours lives elsewhere.
"""

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "Data_Processing/FraudData_ValidationSet.csv"
MODEL_OUT = "model.pkl"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "amt", "zip", "lat", "long", "city_pop", "merch_lat", "merch_long",
    "age", "hour", "day", "month",
    "category_food_dining", "category_gas_transport", "category_grocery_net",
    "category_grocery_pos", "category_health_fitness", "category_home",
    "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos",
    "category_travel",
]

CATEGORY_COLUMNS = [c for c in FEATURE_COLUMNS if c.startswith("category_")]


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    print("Class balance before SMOTE:")
    print(y.value_counts())

    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE
    )

    # max_depth=8 chosen deliberately: the notebook's unconstrained RF hit
    # ~99.2% CV accuracy but showed clear overfit signs. max_depth=8 is the
    # same setting used in the notebook's ROC-AUC comparison, where it
    # achieved 0.986 AUC with much better generalization.
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    print("Training RandomForestClassifier(max_depth=8) ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "categories": CATEGORY_COLUMNS}, MODEL_OUT)
    print(f"Saved model + metadata to {MODEL_OUT}")


if __name__ == "__main__":
    main()
