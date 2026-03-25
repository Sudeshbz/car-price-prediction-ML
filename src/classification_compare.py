import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


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


# ==============================
# 1) Veri setini oku
# ==============================
df = pd.read_csv("../data/CarPrice_Assignment.csv")
print("İlk 5 satır:")
print(df.head())
print("\nVeri seti boyutu:", df.shape)

# ==============================
# 2) Ön temizlik
# ==============================
df = clean_car_name(df)

if "car_ID" in df.columns:
    df = df.drop(columns=["car_ID"])

# ==============================
# 3) Regression yerine classification hedefi oluştur
# price -> düşük / yüksek
# ==============================
median_price = df["price"].median()
df["price_class"] = (df["price"] > median_price).astype(int)

print(f"\nMedian price: {median_price}")
print("Sınıf dağılımı:")
print(df["price_class"].value_counts())

# price artık hedefte kullanılacağı için feature'lardan çıkar
X = df.drop(columns=["price", "price_class"])
y = df["price_class"]

# ==============================
# 4) Sayısal / kategorik sütunlar
# ==============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nSayısal sütunlar:")
print(numeric_features)

print("\nKategorik sütunlar:")
print(categorical_features)

# ==============================
# 5) Ön işleme
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

# GaussianNB sparse matrix ile sıkıntı çıkarabileceği için dense dönüşüm lazım
class DenseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X

# ==============================
# 6) Train / Test ayır
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nEğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

# ==============================
# 7) Modeller
# ==============================
models = {
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "LSVM": LinearSVC(random_state=42, max_iter=5000),
    "RBF SVM": SVC(kernel="rbf", probability=False, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42
    )
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )

results = []

# ==============================
# 8) Eğitim ve değerlendirme
# ==============================
for model_name, model in models.items():
    print(f"\n{model_name} modeli eğitiliyor...")

    if model_name == "Naive Bayes":
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("dense", DenseTransformer()),
            ("model", model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    results.append({
        "Model": model_name,
        "Acc": acc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    })

    print(f"{model_name} sonuçları:")
    print(f"Acc      : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1       : {f1:.4f}")

# ==============================
# 9) Sonuç tablosu
# ==============================
results_df = pd.DataFrame(results).sort_values(by="Acc", ascending=False)

print("\nModel Karşılaştırma Tablosu:")
print(results_df)

# İstersen CSV olarak kaydet
results_df.to_csv("classification_model_comparison.csv", index=False, encoding="utf-8-sig")
print("\nSonuçlar classification_model_comparison.csv dosyasına kaydedildi.")

# ==============================
# 10) Grafik - Accuracy karşılaştırması
# ==============================
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["Acc"])
plt.title("Modellerin Accuracy Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==============================
# 11) Grafik - F1 karşılaştırması
# ==============================
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["F1"])
plt.title("Modellerin F1 Score Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()