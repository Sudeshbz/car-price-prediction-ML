import joblib
import pandas as pd


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
    model_path = "models/best_model.pkl"

    # Modeli yükle
    pipeline = joblib.load(model_path)
    print("Model yüklendi!")

    # Yeni araç verisi
    new_data = pd.DataFrame([
        {
            "CarName": "toyota corolla",
            "symboling": 0,
            "fueltype": "gas",
            "aspiration": "std",
            "doornumber": "four",
            "carbody": "sedan",
            "drivewheel": "fwd",
            "enginelocation": "front",
            "wheelbase": 97.0,
            "carlength": 172.0,
            "carwidth": 65.4,
            "carheight": 52.5,
            "curbweight": 2300,
            "enginetype": "ohc",
            "cylindernumber": "four",
            "enginesize": 130,
            "fuelsystem": "mpfi",
            "boreratio": 3.35,
            "stroke": 3.47,
            "compressionratio": 9.0,
            "horsepower": 111,
            "peakrpm": 5000,
            "citympg": 26,
            "highwaympg": 35
        }
    ])

    # train.py ile aynı ön işleme
    new_data = clean_car_name(new_data)

    print("\nTahmin yapılıyor...")
    predicted_price = pipeline.predict(new_data)[0]

    print(f"\nTahmini araç fiyatı: {predicted_price:.2f}")


if __name__ == "__main__":
    main()