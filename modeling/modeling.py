import os
import mlflow
import mlflow.sklearn
import joblib

import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

from preprocessing.preprocessing import preprocess_data


def train_model(data=None):

    # =========================
    # PATH SETUP
    # =========================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models_dir = os.path.join(BASE_DIR, "models")
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # =========================
    # MLFLOW SETUP (FIX UTAMA)
    # =========================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("carbon_emission_clustering")

    # =========================
    # LOAD DATA
    # =========================
    if data is None:
        data = preprocess_data()

    X = data[['productivity', 'emissions_scaled', 'renewable_supply']]

    # =========================
    # START MLflow RUN
    # =========================
    with mlflow.start_run(run_name="carbon_clustering_v1"):

        print("\n🚀 MLflow RUN STARTED")

        # =========================
        # KMEANS TUNING
        # =========================
        best_k = 2
        best_score = -1
        best_kmeans = None

        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            score = silhouette_score(X, labels)
            print(f"K={k} | silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans

        data["kmeans_cluster"] = best_kmeans.labels_

        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("kmeans_silhouette", best_score)
        mlflow.log_metric("kmeans_inertia", best_kmeans.inertia_)

        # =========================
        # HDBSCAN
        # =========================
        hdb = HDBSCAN(min_cluster_size=5, min_samples=5)
        hdb_labels = hdb.fit_predict(X)
        data["hdbscan_cluster"] = hdb_labels

        n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        mlflow.log_param("hdbscan_clusters", n_clusters_hdb)

        # =========================
        # AGGLOMERATIVE
        # =========================
        agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
        agg_labels = agg.fit_predict(X)

        if len(set(agg_labels)) > 1:
            agg_score = silhouette_score(X, agg_labels)
        else:
            agg_score = 0

        data["final_cluster"] = agg_labels

        mlflow.log_param("agg_n_clusters", 3)
        mlflow.log_metric("agg_silhouette", agg_score)

        # =========================
        # SAVE FILES
        # =========================
        output_file = os.path.join(artifacts_dir, "clustered_result.csv")
        data.to_csv(output_file, index=False)

        joblib.dump(best_kmeans, os.path.join(models_dir, "kmeans.pkl"))
        joblib.dump(agg, os.path.join(models_dir, "agglomerative.pkl"))
        joblib.dump(hdb, os.path.join(models_dir, "hdbscan.pkl"))

        # =========================
        # LOG MLFLOW
        # =========================
        mlflow.sklearn.log_model(best_kmeans, "kmeans_model")
        mlflow.sklearn.log_model(agg, "agg_model")

        mlflow.log_artifact(output_file)

        # =========================
        # TAGGING (BIAR TA BAGUS)
        # =========================
        mlflow.set_tag("project", "carbon_emission_analysis")
        mlflow.set_tag("author", "Faalih")
        mlflow.set_tag("type", "clustering")

        print("\n✅ MLflow logging selesai")
        print("👉 Buka: http://127.0.0.1:5000")

    return best_kmeans, agg, hdb, data


if __name__ == "__main__":
    train_model()