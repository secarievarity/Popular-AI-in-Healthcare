from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'final_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

try:
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)

    # Fitur sesuai dataset kamu (6 kolom)
    FEATURE_NAMES = ['age', 'sex', 'trestbps', 'chol', 'weight', 'height']

    print("Model dan Scaler berhasil dimuat.")

except Exception as e:
    print(f"🚨 Error saat memuat model atau scaler: {e}")
    MODEL = None
    SCALER = None


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST' and MODEL is not None:
        try:
            # Ambil input dari form HTML
            data = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['weight']),
                float(request.form['height'])
            ]

            # Convert ke DataFrame
            input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

            # Scaling
            input_scaled = SCALER.transform(input_df)

            # Prediksi
            result = MODEL.predict(input_scaled)

            prediction = "Tinggi Risiko Penyakit Jantung" if result[0] == 1 else "Rendah Risiko Penyakit Jantung"

        except Exception as e:
            prediction = f"Error Pemrosesan Input: {e}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    print("Server berjalan secara lokal di http://127.0.0.1:5000/")
    app.run(debug=True)
    