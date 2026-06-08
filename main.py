import os
import sys
import traceback

# Import module 
from preprocessing.preprocessing import preprocess_data
from modeling.modeling import train_model


def print_step(step):
    print("\n" + "="*60)
    print(f"🚀 {step}")
    print("="*60)


def main():
    try:
        print_step("START PIPELINE CARBON EMISSION PROJECT")

        # =========================
        # 1. PREPROCESSING
        # =========================
        print_step("1. PREPROCESSING DATA")

        data = preprocess_data()

        if data is None or getattr(data, "empty", False):
            raise ValueError("Preprocessing gagal: data kosong")

        print(" Preprocessing selesai")
        print(f" Shape data: {data.shape}")

        # =========================
        # 2. MODELING
        # =========================
        print_step("2. TRAINING MODEL")

        model, labels = train_model(data)

        print(" Training selesai")
        print(f" Jumlah cluster/label: {len(set(labels))}")

        # =========================
        # 3. SAVE OUTPUT
        # =========================
        print_step("3. SIMPAN HASIL")

        os.makedirs("output", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # simpan hasil clustering
        data["cluster"] = labels
        output_path = "output/hasil_cluster.csv"
        data.to_csv(output_path, index=False)

        print(f" Hasil disimpan di: {output_path}")

        # =========================
        # 4. SAVE MODEL
        # =========================
        import joblib

        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        print(f" Model disimpan di: {model_path}")

        # =========================
        # 5. FINISH
        # =========================
        print_step("PIPELINE SELESAI")
        print("🎉 Semua proses berhasil dijalankan!")
        print("👉 Output siap untuk laporan TA")

    except Exception as e:
        print("\n ERROR TERJADI!")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()