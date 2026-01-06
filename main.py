import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
EXPERIMENT = 3  # 1 = momente, 2 = PCA, 3 = momente + PCA

DATASET_DIR = r"C:\Users\adria\Desktop\SIVA AN1\S1\CPPSMS\Poze"  
IMG_SIZE = (64, 64)              
TEST_SIZE = 0.2
RANDOM_STATE = 42

PCA_COMPONENTS = 50              
KNN_K = 5                        


# -----------------------------
# Utils
# -----------------------------
def list_images_by_folder(dataset_dir: str):
    """
    Returnează o listă de (image_path, label) unde label = numele folderului.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Nu găsesc dataset_dir: {dataset_dir}")

    items = []
    for class_dir in sorted([p for p in dataset_path.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in class_dir.glob("*.jpg"):
            items.append((str(img_path), label))

        

    if len(items) == 0:
        raise RuntimeError("Nu am găsit imagini .jpg în folderele de clase.")
    return items


def load_and_preprocess_image(img_path: str, size=(64, 64)):
    """
    Citește imaginea, o face grayscale, resize, normalizează [0,1].
    Returnează array float32 cu shape (H,W).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Eroare la citire imagine: {img_path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def statistical_moments(img_2d: np.ndarray):
    """
    Momente statistice ale intensităților (pe pixeli).
    Returnează: mean, var, skewness, kurtosis (kurtosis 'excess' sau simplu - aici simplu).
    """
    x = img_2d.flatten().astype(np.float64)
    mu = x.mean()
    var = x.var()

    # evită împărțirea la 0
    sigma = np.sqrt(var) + 1e-12
    skew = np.mean(((x - mu) / sigma) ** 3)
    kurt = np.mean(((x - mu) / sigma) ** 4)

    return np.array([mu, var, skew, kurt], dtype=np.float32)


def build_features(dataset_items, img_size=(64, 64)):
    """
    Construiește:
      - X_img: vectorizarea imaginii (pt PCA)
      - X_mom: momente statistice (pt analiză + eventual clasificare)
      - y: labels
    """
    X_img = []
    X_mom = []
    y = []

    for path, label in dataset_items:
        img = load_and_preprocess_image(path, img_size)
        X_img.append(img.flatten())          # pt PCA (covarianță)
        X_mom.append(statistical_moments(img))
        y.append(label)

    X_img = np.vstack(X_img).astype(np.float32)
    X_mom = np.vstack(X_mom).astype(np.float32)
    y = np.array(y)

    return X_img, X_mom, y


# -----------------------------
# Main
# -----------------------------
def main():
    items = list_images_by_folder(DATASET_DIR)
    df = pd.DataFrame(items, columns=["path", "label"])

    print("Număr total imagini:", len(df))
    print("\nDistribuție pe clase:")
    print(df["label"].value_counts())

    # split stratificat
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # features
    X_img_train, X_mom_train, y_train = build_features(train_df.values.tolist(), IMG_SIZE)
    X_img_test,  X_mom_test,  y_test  = build_features(test_df.values.tolist(), IMG_SIZE)

    # -------------------------
    # PCA pe imaginile vectorizate
    # PCA implică "covariance matrix computation" 
    # -------------------------
    scaler_img = StandardScaler(with_mean=True, with_std=True)
    X_img_train_std = scaler_img.fit_transform(X_img_train)
    X_img_test_std  = scaler_img.transform(X_img_test)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca_train = pca.fit_transform(X_img_train_std)
    X_pca_test  = pca.transform(X_img_test_std)

    print(f"\nPCA: {PCA_COMPONENTS} componente")
    print("Explained variance ratio (primele 10):", np.round(pca.explained_variance_ratio_[:10], 4))
    print("Explained variance total:", float(np.sum(pca.explained_variance_ratio_)))


    # -------------------------
    # Selectare experiment (features)
    # -------------------------
    if EXPERIMENT == 1:
        # EXP1: k-NN doar pe momente
        scaler_mom = StandardScaler()
        X_feat_train = scaler_mom.fit_transform(X_mom_train)
        X_feat_test  = scaler_mom.transform(X_mom_test)
        exp_name = "EXP1: momente"

    elif EXPERIMENT == 2:
        # EXP2: k-NN doar pe PCA
        X_feat_train = X_pca_train
        X_feat_test  = X_pca_test
        exp_name = "EXP2: PCA"

    elif EXPERIMENT == 3:
        # EXP3: k-NN pe [momente + PCA]
        scaler_mom = StandardScaler()
        X_mom_train_std = scaler_mom.fit_transform(X_mom_train)
        X_mom_test_std  = scaler_mom.transform(X_mom_test)

        X_feat_train = np.hstack([X_mom_train_std, X_pca_train])
        X_feat_test  = np.hstack([X_mom_test_std,  X_pca_test])
        exp_name = "EXP3: momente + PCA"

    else:
        raise ValueError("EXPERIMENT trebuie să fie 1, 2 sau 3")

    # -------------------------
    # k-NN
    # -------------------------
    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
    knn.fit(X_feat_train, y_train)
    y_pred = knn.predict(X_feat_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== EXP3: k-NN (k={KNN_K}) pe [momente + PCA] ===")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["label"].unique()))
    print("\nConfusion matrix (ordine etichete alfabetică):")
    print(cm)

    # plot explained variance (util pt articol)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Număr componente PCA")
    plt.ylabel("Varianță cumulată explicată")
    plt.title("PCA - varianță cumulată explicată")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
