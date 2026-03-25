import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


# ==============================
# 1) Veri setini oku
# ==============================
file_path = "CarPrice_Assignment.csv"   # Gerekirse dosya adını değiştir
df = pd.read_csv(r"C:\Users\sudef\Desktop\car-price-project\data\CarPrice_Assignment.csv")

print("İlk 5 satır:")
print(df.head())
print("\nVeri seti boyutu:", df.shape)

# ==============================
# 2) Hedef değişken
# ==============================
target_column = "price"

# Gereksiz ID sütunu varsa çıkar
if "car_ID" in df.columns:
    df = df.drop(columns=["car_ID"])

# X ve y ayır
X = df.drop(columns=[target_column])
y = df[target_column]

# Sayısal / kategorik sütunları bul
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nSayısal sütunlar:")
print(numeric_features)

print("\nKategorik sütunlar:")
print(categorical_features)

# ==============================
# 3) Ön işleme
# ==============================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==============================
# 4) Train / Test ayır
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nEğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

# ==============================
# 5) Modeller
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

if xgb_available:
    models["XGBoost"] = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror"
    )

results = []
trained_pipelines = {}

# ==============================
# 6) Eğitim ve değerlendirme
# ==============================
for model_name, model in models.items():
    print(f"\n{model_name} modeli eğitiliyor...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

    trained_pipelines[model_name] = pipeline

    print(f"{model_name} sonuçları:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 : {r2:.4f}")

# ==============================
# 7) Sonuç tablosu
# ==============================
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

print("\nModel Karşılaştırma Tablosu:")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print(f"\nEn başarılı model: {best_model_name}")

# ==============================
# 8) Grafik 1: Model R2 karşılaştırması
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Modellerin R2 Skorlarının Karşılaştırılması")
plt.xlabel("Model")
plt.ylabel("R2 Skoru")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==============================
# 9) Grafik 2: Model MAE karşılaştırması
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Modellerin MAE Değerlerinin Karşılaştırılması")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==============================
# 10) Grafik 3: Gerçek vs Tahmin
# ==============================
best_predictions = best_pipeline.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_predictions)
plt.title(f"Gerçek Fiyat vs Tahmin Edilen Fiyat ({best_model_name})")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")

min_val = min(y_test.min(), best_predictions.min())
max_val = max(y_test.max(), best_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.tight_layout()
plt.show()

# ==============================
# 11) Feature Importance
# ==============================
model_obj = best_pipeline.named_steps["model"]
preprocessor_fitted = best_pipeline.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()

if hasattr(model_obj, "feature_importances_"):
    importances = model_obj.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nEn önemli ilk 10 özellik:")
    print(importance_df.head(10))

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"].head(10)[::-1],
        importance_df["Importance"].head(10)[::-1]
    )
    plt.title(f"En Önemli 10 Özellik ({best_model_name})")
    plt.xlabel("Önem Skoru")
    plt.ylabel("Özellik")
    plt.tight_layout()
    plt.show()

else:
    print(f"\n{best_model_name} modeli feature importance desteklemiyor.")