# 🚗 Araç Fiyat Tahmini ve Sınıflandırma Uygulaması

## 📊 Veri Seti Özeti

Bu projede kullanılan veri seti, araçlara ait teknik özellikler, fiziksel özellikler ve kategorik bilgilerden oluşmaktadır. Amaç, bu değişkenlerden yararlanarak araç fiyatını tahmin etmek ve araçları fiyat seviyelerine göre sınıflandırmaktır.

Veri seti, araç fiyatını etkileyebilecek birçok faktörü içerdiği için hem regresyon hem de sınıflandırma problemleri için uygun bir yapı sunmaktadır.

---

## 📌 Genel Bilgiler

- Satır sayısı: 205  
- Sütun sayısı: 26  
- Hedef değişken: `price`  
- Problem türü: Regresyon + İkili sınıflandırma  
- Eksik veri: Yok  

---

## 🎯 Hedef Değişken

Bu projede hedef değişken `price` sütunudur.

Sınıflandırma problemi için:

- `0` → Düşük fiyat  
- `1` → Yüksek fiyat  

---

## 🧹 Veri Ön İşleme

- `car_ID` kaldırıldı  
- `CarName → brand` dönüştürüldü  
- Kategorik veriler encode edildi  
- Sayısal veriler ölçeklendi  
- Veri %80 eğitim / %20 test olarak ayrıldı  

---

## 🤖 Modeller ve Sonuçlar

### 🔹 Random Forest (Regresyon)

- R²: **0.9547**  
- MAE: **1323.81**  
- RMSE: **1891.75**

✔ Yüksek doğruluk ile fiyat tahmini yapmaktadır.

---

### 🔹 Classification Modelleri

| Model | Accuracy | Precision | Recall | F1-score |
|------|---------|----------|--------|---------|
| kNN | 0.9268 | 0.8696 | 1.0000 | 0.9302 |
| LSVM | 0.9268 | 0.8696 | 1.0000 | 0.9302 |
| RBF SVM | 0.9268 | 0.8696 | 1.0000 | 0.9302 |
| Random Forest | 0.9268 | 0.8696 | 1.0000 | 0.9302 |
| XGBoost | 0.9268 | 0.8696 | 1.0000 | 0.9302 |
| MLP | 0.9024 | 0.8333 | 1.0000 | 0.9091 |
| Naive Bayes | 0.8049 | 0.7727 | 0.8500 | 0.8095 |

---

## 📊 Genel Yorum

- Çoğu model **%92.6 doğruluk** ile benzer performans göstermektedir  
- **Recall = 1.0** → yüksek fiyatlı araçlar kaçırılmamaktadır  
- En düşük performans: **Naive Bayes**  
- En iyi modeller: **Random Forest, SVM, XGBoost**

---

## 🏁 Sonuç

Bu çalışmada araç fiyat tahmini ve sınıflandırma problemleri başarıyla çözülmüştür. Random Forest modeli hem regresyon hem de sınıflandırma problemlerinde güçlü performans göstermiştir.

Makine öğrenmesi yöntemlerinin gerçek dünya problemlerinde etkili olduğu görülmüştür.

---

## 🛠 Kullanılan Teknolojiler

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- XGBoost  

