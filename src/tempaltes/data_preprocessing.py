import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 
import os 
import warnings
warnings.filterwarnings('ignore')

# Tentukan PATH relatif ke root folder (di luar src/)
DATA_DIR = '../data/'
SCALER_PATH = '../src/scaler.joblib'
DATA_INPUT_FILE = 'heart_disease_data.csv'

def get_absolute_path(relative_path):
    """Mendapatkan path absolut agar script dapat dijalankan dari mana saja."""
    # os.path.dirname(__file__) = folder src/
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(script_dir, relative_path)

def load_data():
    """Memuat data dari file CSV di folder data/."""
    abs_file_path = get_absolute_path(DATA_DIR + DATA_INPUT_FILE)
    try:
        df = pd.read_csv(abs_file_path)
        print(f"Data '{DATA_INPUT_FILE}' berhasil dimuat. Total baris: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"🚨 Error: File tidak ditemukan di {abs_file_path}")
        print("Pastikan 'heart_disease_data.csv' ada di folder 'data/'.")
        return None

def preprocess_and_split_data(df):
    """Melakukan pembersihan data, splitting, dan feature scaling."""
    if df is None:
        return None, None, None, None

    # Pembersihan Data dan Pemisahan Fitur/Target
    df.dropna(inplace=True)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Splitting: Training (80%) dan Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling (Standarisasi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ubah kembali ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Menyimpan scaler ke src/ (PENTING untuk prediksi Hani)
    joblib.dump(scaler, get_absolute_path(SCALER_PATH))
    print(f"✅ StandardScaler berhasil disimpan di: {get_absolute_path(SCALER_PATH)}")

    return X_train_scaled, X_test_scaled, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Menyimpan data yang sudah diproses ke folder data/."""
    try:
        # Menyimpan data output ke folder data/
        X_train.to_csv(get_absolute_path(DATA_DIR + 'X_train.csv'), index=False)
        X_test.to_csv(get_absolute_path(DATA_DIR + 'X_test.csv'), index=False)
        y_train.to_csv(get_absolute_path(DATA_DIR + 'y_train.csv'), index=False, header=True)
        y_test.to_csv(get_absolute_path(DATA_DIR + 'y_test.csv'), index=False, header=True)
        print("✅ Data train/test yang sudah diskalakan berhasil disimpan di folder 'data/'.")
    except Exception as e:
        print(f"🚨 Error saat menyimpan data: {e}")

if __name__ == '__main__':
    data_df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split_data(data_df)
    if X_train is not None:
        save_processed_data(X_train, X_test, y_train, y_test)
        