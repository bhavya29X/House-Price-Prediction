import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TEST_PRED_FILE = "test_predictions.csv"


# -------------------------
# BUILD PREPROCESSING PIPELINE
# -------------------------
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
    )

    return full_pipeline


# ----------------------------------------------------------
# 1️⃣ IF MODEL DOES NOT EXIST → TRAIN MODEL + SAVE DATASETS
# ----------------------------------------------------------
if not os.path.exists(MODEL_FILE):

    housing = pd.read_csv("housing.csv")

    # Create income categories for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)
        test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Save datasets for later testing
    train_set.to_csv(TRAIN_FILE, index=False)
    test_set.to_csv(TEST_FILE, index=False)

    # Prepare data
    housing_labels = train_set["median_house_value"].copy()
    housing_features = train_set.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save model & pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("🎉 Model trained and saved!")
    print("Train data → train.csv")
    print("Test data  → test.csv")


# ----------------------------------------------------------
# 2️⃣ IF MODEL ALREADY EXISTS → EVALUATE ON TEST DATA
# ----------------------------------------------------------
else:
    print("🔍 Model found! Loading model for evaluation...")

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Load the saved test set
    test_set = pd.read_csv(TEST_FILE)

    test_labels = test_set["median_house_value"].copy()
    test_features = test_set.drop("median_house_value", axis=1)

    # Transform test data
    test_prepared = pipeline.transform(test_features)

    # Predict
    predictions = model.predict(test_prepared)

    # Calculate RMSE
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)

    print(f"\n📊 TEST SET EVALUATION:")
    print(f"\nModel Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R^2 score: {r2:.4f}")

    # Save predictions
    results = test_features.copy()
    results["actual"] = test_labels
    results["predicted"] = predictions
    results.to_csv(TEST_PRED_FILE, index=False)
    
    output_df = test_features.copy()
    output_df["actual_price"] = test_labels
    output_df["predicted_price"] = predictions
    output_df["abs_error"] = (output_df["actual_price"] - output_df["predicted_price"]).abs()
    output_df["pct_error"] = output_df["abs_error"] / output_df["actual_price"]
    output_df.to_csv("output_df.csv", index=False)

    print("\n📁 Predictions saved as: test_predictions.csv")
    print("✔ Contains: features + actual price + predicted price")
    
    print("\n output_df.csv contains:- features + actual price + predicted price + absolute error + percentage error")





