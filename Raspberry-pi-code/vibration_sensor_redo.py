import smbus2
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import numpy as np

from adc import *


# I2C address of the Seeed 8-channel 12-bit ADC
ADC_ADDRESS = 0x08

# Initialize I2C bus
bus = smbus2.SMBus(1)

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0] * size
        self.index = 0
        self.full = False
        print(f"CircularBuffer initialized with size {size}")

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True
        #print(f"Appended value {value} to buffer at index {self.index}")

    def get(self):
        if self.full:
            return self.buffer[self.index:] + self.buffer[:self.index]
        else:
            return self.buffer[:self.index]

# Function to acquire data and count parts
def acquire_data_and_count(adc, channel, sample_rate, buffer, stop_event, min_index, max_index, amplitude_threshold, time_threshold, fft_buffer, bias):
    global part_count, rolling_counter, duplicate_check
    interval = 1.0 / sample_rate
    start_time = time.perf_counter()
    sample_counter = 0
    print("Starting data acquisition")

    try:
        while not stop_event.is_set():
            current_time = time.perf_counter()
            if current_time - start_time >= interval:
                print(current_time- start_time)
                start_time = current_time
                raw_data = adc.read_voltage(channel) - bias
                buffer.append(raw_data)
                #print(f"Sample {sample_counter}: {raw_data}")

    except KeyboardInterrupt:
        print("Data acquisition stopped")
    finally:
        print("Data acquisition stopped")
        bus.close()

# Counting algorithm
def check_part(fft_data, min_index, max_index, amplitude_threshold, time_threshold):
    global part_count, rolling_counter, duplicate_check
    target_avg = np.mean(fft_data[min_index:max_index])
    #print(f"FFT data average in range {min_index}-{max_index}: {target_avg}")

    if target_avg >= amplitude_threshold:
        if not duplicate_check:  # Only increment if not already counted
            rolling_counter += 1
            print("Rolling counter: ", rolling_counter)
            #print(f"Rolling counter incremented: {rolling_counter}")
            if rolling_counter >= time_threshold:
                part_count += 1
                rolling_counter = 0
                duplicate_check = True
                print(f"Part counted: {part_count}")
    else:
        rolling_counter = 0
        duplicate_check = False  # Reset flag while below threshold

# Function to update the plot
def update_plot(frame, buffer, line):
    data = buffer.get()
    if data:
        line.set_ydata(data)
        line.set_xdata(range(len(data)))
    return line,

# Function to update the FFT plot
def update_plot_fft(frame, buffer, line_fft):
    if len(buffer.get()):
        data = buffer.get()
        data_fft = np.abs(np.fft.fft(data))[:len(data)//2]
        line_fft.set_ydata(data_fft)
        line_fft.set_xdata(range(len(data_fft)))
    return line_fft,

def main():
    global part_count, rolling_counter, duplicate_check
    part_count = 0
    rolling_counter = 0
    duplicate_check = False

    adc = ADC()

    channel = 0  # Specify the channel you want to read from
    bias = 500

    sample_rate = 20000  # Desired sample rate in Hz
    buffer_size = 2000  # Buffer size 
    buffer = CircularBuffer(buffer_size)
    fft_buffer = np.zeros(buffer_size // 2)  # Buffer for FFT data

    min_index = 10  # Example value, adjust as needed
    max_index = 50  # Example value, adjust as needed
    amplitude_threshold = 100  # Example value, adjust as needed
    time_threshold = 5  # Example value, adjust as needed

    print("Nyquist frequency: ", sample_rate / 2)
    print("FFT operates on: ", buffer_size/sample_rate, "seconds of data")
    print("FFT resolution: ", sample_rate/buffer_size, "Hz")

    # Start data acquisition and part counting in a separate thread
    stop_event = threading.Event() # Create stop event
    acquisition_thread = threading.Thread(target=acquire_data_and_count, args=(adc, channel, sample_rate, buffer, stop_event, min_index, max_index, amplitude_threshold, time_threshold, fft_buffer, bias))
    acquisition_thread.start()

    # Calculate corresponding frequency bins for FFT (positive frequencies only)
    frequencies = np.fft.fftfreq(buffer_size, 1/sample_rate)[:buffer_size//2]

    # Set up the plot for time-domain data
    fig1, ax1 = plt.subplots()
    line, = ax1.plot([], [], lw=2)
    ax1.set_ylim(-2000, 2000)  # Adjust based on ADC resolution
    ax1.set_xlim(0, buffer_size)  
    ax1.set_title("Real-Time Vibration Sensor Data")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("ADC Value")

    # Set up the plot for frequency-domain data (FFT)
    fig2, ax2 = plt.subplots()
    line_fft, = ax2.plot([], [])  # Initialize with zeros
    ax2.set_xlim(-10, buffer_size)  # Set x-axis to go from 0 to Nyquist frequency

    # Custom x ticks (mapping data points to actual frequencies)
    num_ticks = 6  # You can adjust this number as needed
    tick_positions = np.linspace(0, len(frequencies)-1, num_ticks).astype(int)  # Select tick positions
    tick_labels = [f'{frequencies[i]:.0f}' for i in tick_positions]  # Map these to frequency values

    # Set x-ticks and labels
    #ax2.set_xticks(frequencies[tick_positions])  # Set the x-ticks to the actual frequency values
    #ax2.set_xticklabels(tick_labels)  # Set the x-tick labels to the frequency values
    ax2.set_ylim(0, 800000)

    # Start the animation
    ani1 = FuncAnimation(fig1, update_plot, fargs=(buffer, line), interval=50, blit=True)
    ani2 = FuncAnimation(fig2, update_plot_fft, fargs=(buffer, line_fft), interval=20, blit=True)
    plt.show()

    # Signal the acquisition thread to stop
    stop_event.set()
    acquisition_thread.join()

if __name__ == "__main__":
    main()

