import csv
from datetime import datetime
import time
import numpy as np
from gpiozero import MCP3202
from scipy.fftpack import fft
from scipy.stats import entropy
import joblib  # saving and loading model
from sklearn.cluster import KMeans


def convert_to_buffers(arr, buffer_size: int):
    buffers = []
    for i in range(0, len(arr) - buffer_size, buffer_size):
        buffers.append(arr[i:i + buffer_size])
    return buffers


def extract_features(buffer, fs):
    N = len(buffer)
    freqs = np.fft.rfftfreq(N, 1 / fs)  # get frequency bins
    spectrum = np.abs(fft(buffer)[:len(freqs)])  # compute fft magnitude

    # compute features
    spectral_energy = np.sum(spectrum[(freqs >= 4000) & (freqs <= 6000)] ** 2)
    peak_freq = freqs[np.argmax(spectrum)]
    spectral_entropy = entropy(spectrum / np.sum(spectrum))
    rms_amplitude = np.sqrt(np.mean(np.array(buffer) ** 2))

    features = [spectral_energy, peak_freq, spectral_entropy, rms_amplitude]

    return features 


def identify_labels(kmeans_model, feature_vectors, labels):
    # Calculate average spectral energy for each cluster
    unique_clusters = np.unique(labels)
    
    cluster_energies = [
        np.mean([fv[0] for i, fv in enumerate(feature_vectors) if labels[i] == c])
        for c in unique_clusters
    ]
    
    """
    cluster_rms = [
        np.mean([fv[3] for i, fv in enumerate(feature_vectors) if labels[i] == c])
        for c in unique_clusters
    ]
    """
    
    print(cluster_energies)
    
    grinding_cluster = unique_clusters[np.argmax(cluster_energies)]

    return grinding_cluster


def classify_vibration(signal, kmeans_model, grinding_cluster, fs):
    # extract features from incoming vibration data
    feature_vector = extract_features(signal, fs)

    # predict cluster
    cluster = kmeans_model.predict([feature_vector])[0]

    print(int(cluster))
    # return grinding detection
    return int(cluster == grinding_cluster)


if __name__ == '__main__':
    try:
        sample_counter = 0
        voltage_data = []
        pot = MCP3202(channel=0)
        print("-----Starting data acquisition-----")
        start_time = time.time()

        while True:
            sample_counter += 1
            voltage = pot.value
            voltage_data.append(voltage)

    except KeyboardInterrupt:
        print("-----Program interrupted by user-----")

    finally:
        # Signal the acquisition thread to stop
        elapsed_time = time.time() - start_time
        sample_rate = sample_counter / elapsed_time
        print("-----Data acquisition stopped-----")
        print(f"Time elapsed: {elapsed_time:.5f} seconds")
        print(f"Sample rate: {sample_counter / elapsed_time:.5f} samples per second")

        # Train
        print("---Beginning training---")
        # Convert into buffers
        buffer_size = 500
        amplitude_buffers = convert_to_buffers(voltage_data, buffer_size)
        print("Buffered")

        # Extract features
        feature_vectors = np.array([extract_features(buffer, round(sample_rate)) for buffer in amplitude_buffers])
        print("Extracted")

        # Train k-means
        kmeans_model = KMeans(n_clusters=3, random_state=42)
        kmeans_model.fit(feature_vectors)  # Fit the model
        labels = kmeans_model.labels_  # Get the labels from the model

        # Identify grinding cluster
        grinding_cluster = identify_labels(kmeans_model, feature_vectors, labels)
        print(f"Grinding cluster: {grinding_cluster}")
        print("Labeled")

        # Save model and cluster label
        joblib.dump(kmeans_model, "kmeans_model.pkl")
        joblib.dump(grinding_cluster, "grinding_cluster.pkl")

        print("Model training complete. Model saved")
