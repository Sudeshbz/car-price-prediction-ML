import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def clean_car_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "CarName" in df.columns:
        df["brand"] = df["CarName"].astype(str).str.split().str[0].str.lower()

        brand_corrections = {
            "maxda": "mazda",
            "porcshce": "porsche",
            "toyouta": "toyota",
            "vokswagen": "volkswagen",
            "vw": "volkswagen",
        }

        df["brand"] = df["brand"].replace(brand_corrections)

    return df


def main():
    data_path = os.path.join("data", "CarPrice_Assignment.csv")
    model_path = os.path.join("models", "best_model.pkl")
    metrics_path = os.path.join("outputs", "metrics.txt")
    feature_plot_path = os.path.join("outputs", "feature_importance.png")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Veri okunuyor..."),
    data_path = r"C:\Users\sudef\Desktop\car-price-project\data\CarPrice_Assignment.csv"
    df = pd.read_csv(data_path)
  

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nVeri boyutu:", df.shape)

    # CarName'den brand çıkar
    df = clean_car_name(df)

    # Gereksiz kolon sil
    if "car_ID" in df.columns:
        df = df.drop(columns=["car_ID"])

    target = "price"

    if target not in df.columns:
        raise ValueError("Dataset içinde 'price' sütunu bulunamadı.")

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nSayısal kolonlar:")
    print(numeric_features)

    print("\nKategorik kolonlar:")
    print(categorical_features)

    # Ön işleme
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train / Test ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\nModel eğitiliyor...")
    pipeline.fit(X_train, y_train)

    print("Tahmin yapılıyor...")
    y_pred = pipeline.predict(X_test)

    # Metrikler
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Sonuçları ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")

    # metrics.txt kaydet
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Araç Fiyat Tahmini - Model Sonuçları\n")
        f.write("-----------------------------------\n")
        f.write(f"R2 Score : {r2:.4f}\n")
        f.write(f"MAE      : {mae:.2f}\n")
        f.write(f"RMSE     : {rmse:.2f}\n")

    print(f"\nMetrikler kaydedildi: {metrics_path}")

    # Model kaydet
    joblib.dump(pipeline, model_path)
    print(f"Model kaydedildi: {model_path}")

    # Feature importance grafiği
    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    model_fitted = pipeline.named_steps["model"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = model_fitted.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top_n = 15
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Özellik Önemi")
    plt.tight_layout()
    plt.savefig(feature_plot_path, dpi=200)
    plt.close()

    print(f"Özellik önem grafiği kaydedildi: {feature_plot_path}")


if __name__ == "__main__":
    main()