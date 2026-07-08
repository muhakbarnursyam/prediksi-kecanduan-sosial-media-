import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# MEMBACA DATASET
# =====================================================

data = pd.read_csv("dataset.csv")

print("=" * 60)
print("5 DATA PERTAMA")
print("=" * 60)
print(data.head())

print("\nJumlah Data :", len(data))
print("Jumlah Kolom :", len(data.columns))

# =====================================================
# MEMISAHKAN FITUR DAN LABEL
# =====================================================

X = data.drop("label", axis=1)
y = data["label"]

# =====================================================
# MEMBAGI DATA
# 80% Training
# 20% Testing
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nJumlah Data Training :", len(X_train))
print("Jumlah Data Testing  :", len(X_test))

# =====================================================
# MEMBUAT MODEL
# =====================================================

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    max_depth=5
)

# =====================================================
# TRAINING
# =====================================================

model.fit(X_train, y_train)

print("\nTraining selesai.")

# =====================================================
# PREDIKSI
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# AKURASI
# =====================================================

akurasi = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)

print(f"Akurasi Model : {akurasi*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =====================================================
# SIMPAN MODEL
# =====================================================

joblib.dump(model, "model.pkl")

print("\nModel berhasil disimpan dengan nama model.pkl")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Fitur": X.columns,
    "Nilai": model.feature_importances_
})

importance = importance.sort_values(
    by="Nilai",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# =====================================================
# GRAFIK FEATURE IMPORTANCE
# =====================================================

plt.figure(figsize=(8,5))

plt.bar(
    importance["Fitur"],
    importance["Nilai"]
)

plt.title("Feature Importance")
plt.xlabel("Fitur")
plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()

# =====================================================
# CONFUSION MATRIX
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Confusion Matrix")

plt.show()

print("\nProgram selesai.")