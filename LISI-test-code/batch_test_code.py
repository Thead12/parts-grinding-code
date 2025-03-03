#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
# Copyright (C) 2018  Seeed Technology Co.,Ltd. 
#
# This is the ADC library for Grove Base Hat
# which used to connect grove sensors for raspberry pi.
# 

import sys
import i2c
import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
import numpy as np

from smbus2 import SMBus
from CircularBuffer import CircularBuffer


def convert_to_voltage(block):
    for i in range(2, len(block)-1, 2):
        voltage = block[i] << 8 | block[i+1]
        return (voltage - (1650+20000))

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

# Function to acquire data and count parts
def acquire_data_and_count(channel, buffer, voltage_data, time_data, stop_event):
    start_time = time.time()
    sample_counter = 0
    print("Starting data acquisition")

    try:
        while not stop_event.is_set():
            sample_counter += 8
            block = bus.read_i2c_block_data(0x08, 0x20, 32) # obtain data
            voltage = convert_to_voltage(block)             # convert data
            buffer.append(voltage)                          # store data
            voltage_data.append(voltage)
            time_data.append(time.time() - start_time)

    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
        elapsed_time = time.time() - start_time
        print("Data acquisition stopped")
        print(f"Time elapsed: {elapsed_time:.5f} seconds")
        print(f"Sample rate: {sample_counter / elapsed_time:.5f} samples per second")
        
        # Generate a unique filename
        filename = datetime.now().strftime("vibration_data_%Y%m%d_%H%M%S.csv")
        
        # Write data to CSV file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Voltage"])
            for t, v in zip(time_data, voltage_data):
                writer.writerow([t, v])
        

if __name__ == '__main__':
    # I2C address of the Seeed 8-channel 12-bit ADC
    ADC_ADDRESS = 0x08
    # Initialize I2C bus
    bus = SMBus(1)
    channel = 0
    
    buffer = CircularBuffer(1000)

    # Start data acquisition and part counting in a separate thread
    stop_event = threading.Event()  # Create stop event
    acquisition_thread = threading.Thread(target=acquire_data_and_count, args=(channel, buffer, stop_event))
    acquisition_thread.start()

    # Set up the plot for time-domain data
    fig1, ax1 = plt.subplots()
    line1, = ax1.plot([], [], lw=2)
    ax1.set_ylim(-3000, 100000)  # Adjust based on ADC resolution
    ax1.set_xlim(0, 1000)  
    ax1.set_title("Real-Time Vibration Sensor Data")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("ADC Value")

    # Set up the plot for frequency-domain data
    fig2, ax2 = plt.subplots()
    line2, = ax2.plot([], [], lw=2)
    ax2.set_ylim(0, 200000)  # Adjust based on FFT result
    ax2.set_xlim(0, 1.5)  # FFT frequency range
    ax2.set_title("Frequency Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")

    # Start the animations
    ani1 = FuncAnimation(fig1, update_time_plot, fargs=(buffer, line1), interval=50, blit=True)
    ani2 = FuncAnimation(fig2, update_freq_plot, fargs=(buffer, line2), interval=50, blit=True)
    plt.show()

    # Signal the acquisition thread to stop
    stop_event.set()
    acquisition_thread.join()






