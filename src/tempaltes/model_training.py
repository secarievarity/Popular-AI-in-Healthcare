import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '../data/dataset.csv'   # ganti sesuai nama file aslinya
MODEL_PATH = '../src/final_model.joblib'
SCALER_PATH = '../src/scaler.joblib'
FEATURE_PATH = '../src/feature_names.joblib'


def get_absolute_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_dataset():
    try:
        df = pd.read_csv(get_absolute_path(DATA_PATH))

        # 👉 Pastikan hanya pakai fitur dataset kamu
        FEATURE_NAMES = ['age', 'sex', 'trestbps', 'chol', 'weight', 'height']

        X = df[FEATURE_NAMES]
        y = df['target']   # pastikan ini nama kolom label

        print("Dataset berhasil dimuat.")
        return X, y, FEATURE_NAMES

    except Exception as e:
        print(f"🚨 Error load dataset: {e}")
        return None, None, None


def train_model(X, y, FEATURE_NAMES):

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=8
    )

    print("\n--- Mulai training model ---")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\nAkurasi Testing:", round(acc * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # simpan model
    joblib.dump(model, get_absolute_path(MODEL_PATH))
    joblib.dump(scaler, get_absolute_path(SCALER_PATH))
    joblib.dump(FEATURE_NAMES, get_absolute_path(FEATURE_PATH))

    print("\n✅ Model, scaler, dan feature names berhasil disimpan!")
    return model


if __name__ == '__main__':
    X, y, FEATURE_NAMES = load_dataset()
    if X is not None:
        train_model(X, y, FEATURE_NAMES)
