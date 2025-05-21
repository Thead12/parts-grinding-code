import sys
import csv
from datetime import datetime
import time
import threading
import numpy as np
from gpiozero import MCP3202, LED
import joblib
from CircularBuffer import CircularBuffer
import queue
from collect_and_train import extract_features

# Function to acquire data and count parts
def acquire_data(buffer, stop_event, data_queue, buffer_size):
    sample_counter = 0
    pot = MCP3202(channel=0)
    print("-----Starting data acquisition-----")
    start_time = time.time()

    try:
        while not stop_event.is_set():  # Check if stop_event is set
            sample_counter += 1
            voltage = pot.value - 0.35  # Offset for the sensor
            buffer.append(voltage)  # Append data to the circular buffer
            # After appending a new sample, check if the buffer is full
            if sample_counter == buffer_size:
                # Pass the buffer data to the classification thread through the queue
                data_queue.put(buffer.get())  # Put the entire buffer into the queue
                sample_counter = 0  # Reset counter after sending buffer

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("-----Data acquisition stopped-----")
        print(f"Time elapsed: {elapsed_time:.5f} seconds")
        print(f"Sample rate: {sample_counter / elapsed_time:.5f} samples per second")

# Classification function
def classify_vibration(features, kmeans_model, grinding_cluster):
    """
    Classifies whether a given vibration signal is grinding or idle.
    """
    # Predict which cluster this feature belongs to
    cluster = kmeans_model.predict([features])[0]
    return int(cluster == grinding_cluster)  # 1 = Grinding, 0 = Idle

# Function to handle classification in a separate thread
def classify_data(data_queue, kmeans_model, grinding_cluster):
    voltage_output = LED(17)
    try:
        grinding_count = 0
        was_grinding = False
        grinding_for = 0
        not_grinding_for = 0
        while True:
            # Get data from the queue for classification
            data = data_queue.get()  # Blocks until there's data
            if data is None:  # Stop signal from the main thread
                break
        
            # Extract features from the vibration data
            features = extract_features(data, 10000)
                    
            # Classify the current signal
            is_grinding = classify_vibration(features, kmeans_model, grinding_cluster)
            
            # Handle classification result
            if is_grinding:
                print("Grinding detected!")
                grinding_for+=1
                if not was_grinding and grinding_for >= 30:
                    voltage_output.on()
                    not_grinding_for = 0
                    #voltage_output.on()
                    grinding_count += 1
                    print(f"Part count: {grinding_count}")
                    grinding_for = 0
                    was_grinding = True
                    #voltage_output.off()
                #elif grinding_for >= 30:
                    #grinding_for = 0
            else:
                print("Machine Idle!")
                not_grinding_for += 1
                if not_grinding_for >=5:
                    voltage_output.off()
                    grinding_for = 0
                    was_grinding = False
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Total grinding process count: {grinding_count}")

if __name__ == '__main__':
    try:
        buffer_size = 700  # Circular buffer size
        buffer = CircularBuffer(buffer_size)

        # Load trained K-Means model and grinding cluster information
        kmeans_model = joblib.load("kmeans_model.pkl")
        grinding_cluster = joblib.load("grinding_cluster.pkl")

        # Create a thread-safe queue to pass data between threads
        data_queue = queue.Queue()

        # Start data acquisition thread
        stop_event = threading.Event()  # Create stop event
        acquisition_thread = threading.Thread(target=acquire_data, args=(buffer, stop_event, data_queue, buffer_size))
        acquisition_thread.start()

        # Start classification thread
        classification_thread = threading.Thread(target=classify_data, args=(data_queue, kmeans_model, grinding_cluster))
        classification_thread.start()

        # Wait for user to interrupt the program
        while True:
            time.sleep(1)  # Keep the main thread alive

    except KeyboardInterrupt:
        print("-----Program interrupted by user-----")
        stop_event.set()  # Signal the acquisition thread to stop
        data_queue.put(None)  # Send stop signal to the classification thread

    finally:
        # Ensure threads are joined properly
        acquisition_thread.join()

        classification_thread.join()
