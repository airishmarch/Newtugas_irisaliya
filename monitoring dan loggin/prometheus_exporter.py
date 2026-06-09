# prometheus_exporter.py
import time
import random
import requests
from prometheus_client import start_http_server, Counter, Histogram, Gauge


# ========================
# DEFINISI METRICS
# ========================
REQUEST_COUNT = Counter(
    'model_request_total',
    'Total request ke model'
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency request ke model',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

SILHOUETTE_SCORE = Gauge(
    'model_silhouette_score',
    'Silhouette score model clustering'
)

ACTIVE_CLUSTERS = Gauge(
    'model_active_clusters',
    'Jumlah cluster aktif (KMeans best_k)'
)

PREDICTION_ERROR = Counter(
    'model_prediction_error_total',
    'Total error saat prediksi'
)

# ========================
# FUNGSI SIMULASI REQUEST
# ========================
def simulate_requests():
    """Simulasi request ke MLflow serve dan catat metrics"""
    
    # Contoh payload sesuai fitur kamu
    payload = {
        "inputs": [[0.5, 0.3, 0.8]]  # [productivity, emissions_scaled, renewable_supply]
    }

    with REQUEST_LATENCY.time():
        try:
            response = requests.post(
                "http://127.0.0.1:5001/invocations",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            REQUEST_COUNT.inc()

            # Update gauge metrics (simulasi nilai)
            SILHOUETTE_SCORE.set(round(random.uniform(0.3, 0.7), 4))
            ACTIVE_CLUSTERS.set(3)  # sesuaikan best_k kamu

        except Exception as e:
            PREDICTION_ERROR.inc()
            print(f"Error: {e}")

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    # Jalankan HTTP server untuk Prometheus scrape di port 8000
    start_http_server(8000)
    print("✅ Prometheus exporter jalan di http://localhost:8000/metrics")

    while True:
        simulate_requests()
        time.sleep(5)  # kirim request tiap 5 detik