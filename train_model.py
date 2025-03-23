import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.stats import entropy
import joblib # saving and loading model
from sklearn.cluster import KMeans


SAMPLE_RATE = 10000

def convert_to_buffers(arr: np.array, buffer_size: int):
    buffers = []
    for i in range(0, arr.size - buffer_size, buffer_size):
        buffers.append(arr[i:i+buffer_size])
    return buffers

def extract_features(signal, fs=SAMPLE_RATE):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs) # get frequency bins
    spectrum = np.abs(fft(signal)[:len(freqs)]) # compute fft magnitude

    # compute features
    spectral_energy = np.sum(spectrum[(freqs >= 4000) & (freqs <= 6000)] ** 2)
    peak_freq = freqs[np.argmax(spectrum)]
    spectral_entropy = entropy(spectrum / np.sum(spectrum))
    rms_amplitude = np.sqrt(np.mean(signal ** 2))

    return [spectral_energy, peak_freq, spectral_entropy, rms_amplitude]

def identify_labels(kmeans_model, feature_vectors, labels):
    # Calculate average spectral energy for each cluster
    unique_clusters = np.unique(labels)
    cluster_energies = [
        np.mean([fv[0] for i, fv in enumerate(feature_vectors) if labels[i] == c])  
        for c in unique_clusters
    ]
    
    # The cluster with the highest spectral energy is likely grinding
    grinding_cluster = unique_clusters[np.argmax(cluster_energies)]
    
    return grinding_cluster

def classify_vibration(signal, kmeans_model, grinding_cluster, fs=SAMPLE_RATE):
    # extract features from incoming vibration data
    feature_vector = extract_features(signal, fs)

    # predict cluster
    cluster = kmeans_model.predict([feature_vector])[0]

    # return grinding detection
    return int(cluster == grinding_cluster)

if __name__ == '__main__':
    # load data
    df = pd.read_csv('data/vibration_data_20250321_140803.csv')
    amplitude_data = df.to_numpy()

    print("Data loaded")
    print(df.head())

    # convert into buffers
    buffer_size = 500
    amplitude_buffers = convert_to_buffers(amplitude_data, buffer_size)
    print("Buffered")

    # extract features
    feature_vectors = np.array([extract_features(sig) for sig in amplitude_buffers])
    print("Extracted")

    # train k-means
    kmeans_model = KMeans(n_clusters=2, random_state=42)
    labels = kmeans_model.fit(feature_vectors)
    print("Trained")

    # identify grinding cluster
    grinding_cluster = identify_labels(kmeans_model, feature_vectors, labels)
    print(f"Grinding cluster: {grinding_cluster}")
    print("labeled")

    # save model and cluster label
    joblib.dump(kmeans_model, "kmeans_model.pkl")
    joblib.dump(grinding_cluster, "grinding_cluster.pkl")

    print("Model training complete. Model saved")

    
