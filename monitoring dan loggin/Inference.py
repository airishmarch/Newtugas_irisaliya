import requests
import json
import pandas as pd

# ========================
# KONFIGURASI
# ========================
MLFLOW_SERVE_URL = "http://127.0.0.1:5001/invocations"

# ========================
# FUNGSI INFERENCE
# ========================
def predict(productivity, emissions_scaled, renewable_supply):
    """
    Melakukan prediksi cluster dari input data.
    
    Parameters:
    - productivity: float
    - emissions_scaled: float
    - renewable_supply: float
    
    Returns:
    - cluster: int (hasil prediksi cluster)
    """
    payload = {
        "inputs": [[productivity, emissions_scaled, renewable_supply]]
    }

    try:
        response = requests.post(
            MLFLOW_SERVE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            cluster = result["predictions"][0]
            return cluster
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Gagal konek ke MLflow serve: {e}")
        return None


# ========================
# CONTOH INFERENSI SINGLE
# ========================
def inference_single():
    print("=" * 50)
    print("🔍 INFERENCE SINGLE DATA")
    print("=" * 50)

    # Contoh input data
    productivity = 0.5
    emissions_scaled = 0.3
    renewable_supply = 0.8

    print(f"Input:")
    print(f"  productivity     : {productivity}")
    print(f"  emissions_scaled : {emissions_scaled}")
    print(f"  renewable_supply : {renewable_supply}")

    cluster = predict(productivity, emissions_scaled, renewable_supply)

    if cluster is not None:
        print(f"\n✅ Hasil Prediksi Cluster: {cluster}")
    else:
        print("\n❌ Prediksi gagal")


# ========================
# CONTOH INFERENSI BATCH
# ========================
def inference_batch():
    print("\n" + "=" * 50)
    print("🔍 INFERENCE BATCH DATA")
    print("=" * 50)

    # Contoh beberapa data sekaligus
    data = [
        {"productivity": 0.2, "emissions_scaled": 0.8, "renewable_supply": 0.1},
        {"productivity": 0.7, "emissions_scaled": 0.2, "renewable_supply": 0.9},
        {"productivity": 0.5, "emissions_scaled": 0.5, "renewable_supply": 0.5},
        {"productivity": 0.9, "emissions_scaled": 0.1, "renewable_supply": 0.7},
    ]

    results = []
    for i, row in enumerate(data):
        cluster = predict(
            row["productivity"],
            row["emissions_scaled"],
            row["renewable_supply"]
        )
        results.append({**row, "predicted_cluster": cluster})
        print(f"Data {i+1}: cluster = {cluster}")

    # Tampilkan sebagai DataFrame
    df = pd.DataFrame(results)
    print("\n📊 Hasil Batch Inference:")
    print(df.to_string(index=False))

    return df


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("🚀 MEMULAI INFERENCE")
    print("Pastikan MLflow serve sudah jalan di http://127.0.0.1:5001\n")

    # Single inference
    inference_single()

    # Batch inference
    inference_batch()

    print("\n✅ Inference selesai!")
