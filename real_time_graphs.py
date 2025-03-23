import sys
import csv
#import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
import numpy as np
from gpiozero import MCP3202

from CircularBuffer import CircularBuffer
   
#from LoadClassifier import * 


# Function to update the time-domain plot
def update_time_plot(frame, buffer, line):
    data = buffer.get()
    if data:
        line.set_ydata(data)
        line.set_xdata(range(len(data)))
    return line,

# Function to update the frequency-domain plot
def update_freq_plot(frame, buffer, line):
    data = buffer.get()
    if data:
        # Compute the FFT
        fft_data = np.fft.fft(data)
        fft_freq = np.fft.fftfreq(len(data))

        line.set_ydata(np.abs(fft_data))
        line.set_xdata(fft_freq)
    return line,

def assign_class_fft(amplitude_fft, min_freq, max_freq, threshold):
    # Assign the class based on the FFT result
    if np.max(amplitude_fft[min_freq:max_freq]) > threshold:
        return 1
    else:
        return 0

# Function to acquire data and count parts
def acquire_data_and_count(buffer, stop_event):
    sample_counter = 0
    voltage_data = []
    time_data = []
    pot = MCP3202(channel=0)
    print("-----Starting data acquisition-----")
    start_time = time.time()

    try:
        while not stop_event.is_set():
            sample_counter += 1
            voltage = pot.value - 0.35
            buffer.append(voltage)   

            # machine learning method
            #prediction = classifier.predict(buffer)
            #print("Prediction: ", prediction)                    
            
            voltage_data.append(voltage)
            time_data.append(time.time()) # - start_time)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("-----Data acquisition stopped-----")
        print(f"Time elapsed: {elapsed_time:.5f} seconds")
        print(f"Estimate Sample rate: {sample_counter / elapsed_time:.5f} samples per second")
        print(f"True Sample rate: {len(voltage_data) / (time_data[-1]-time_data[0]):.5f} samples per second")
        
        
        # Generate a unique filename
        filename = datetime.now().strftime("data/vibration_data_%Y%m%d_%H%M%S.csv")
        
        # Write data to CSV file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Voltage"])
            for t, v in zip(time_data, voltage_data):
                writer.writerow([t, v])
        

if __name__ == '__main__':
    try:      

        #classifier = RealTimeClassifier('binary_classifier.pth', buffer_size=1000)

        buffer_size = 1000
        buffer = CircularBuffer(buffer_size)

        # Start data acquisition and part counting in a separate thread
        stop_event = threading.Event()  # Create stop event
        acquisition_thread = threading.Thread(target=acquire_data_and_count, args=(buffer, stop_event))
        acquisition_thread.start()

        # Set up the plot for time-domain data
        fig1, ax1 = plt.subplots()
        line1, = ax1.plot([], [], lw=2)
        ax1.set_ylim(-0.5, 0.5)  # Adjust based on ADC resolution
        ax1.set_xlim(0, buffer_size)  
        ax1.set_title("Real-Time Vibration Sensor Data")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("ADC Value")

        # Set up the plot for frequency-domain data
        fig2, ax2 = plt.subplots()
        line2, = ax2.plot([], [], lw=2)
        ax2.set_ylim(0, 10.0)  # Adjust based on FFT result
        ax2.set_xlim(0, 0.5)  # FFT frequency range
        ax2.set_title("Frequency Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")

        # Start the animations
        ani1 = FuncAnimation(fig1, update_time_plot, fargs=(buffer, line1), interval=100, blit=True)
        ani2 = FuncAnimation(fig2, update_freq_plot, fargs=(buffer, line2), interval=100, blit=True)
        plt.show()

    except KeyboardInterrupt:
        print("-----Program interrupted by user-----")

    finally:
        # Signal the acquisition thread to stop
        stop_event.set()
        acquisition_thread.join()
        print("-----Acquisition thread stopped-----")


