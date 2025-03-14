import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from matplotlib.animation import FuncAnimation

def convert_to_buffers(arr: np.array, buffer_size: int):
    buffers = []
    for i in range(0, arr.size - buffer_size, buffer_size):
        buffers.append(arr[i:i+buffer_size])
    return buffers

def rms(buffers):
    rms_voltages = []

    for buffer in buffers:
        rms_voltage = np.sqrt(np.mean(buffer**2))
        rms_voltages.append(rms_voltage)
    return rms_voltages

def assign_class_rms(rms_voltages, threshold):
    binary_labels = []  
    for i in range(len(rms_voltages)):
        if np.abs(rms_voltages[i]) > threshold:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    return binary_labels

def assign_class_fft(amplitude_fft, min_freq, max_freq, threshold):
    binary_labels = []
    for i in amplitude_fft:
        mean_freq = np.mean(i[min_freq:max_freq])
        if mean_freq > threshold:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
        #print("Frequency: ", mean_freq, binary_labels[-1])
    
    return binary_labels

def fft_of_buffer(buffers, buffer_size):
    amplitude_ffts = []
    frequencies = []

    for buffer in buffers:
        amplitude_ffts.append(np.abs(np.fft.fft(buffer)[:buffer_size//2]))
        frequencies.append(np.fft.fftfreq(len(buffer), d=buffer_size/1000)[:buffer_size//2])  # Adjust d to match the time interval

    return amplitude_ffts, frequencies

def count_stats(fft_labels, rms_labels):
    # calculate difference between rms method and fft method
    print("Total number of labels: ", len(binary_labels_fft))
    fft_sum = np.sum(binary_labels_fft)
    rms_sum = np.sum(binary_labels_rms)
    print("Sum of fft: {} and  rms: {} labels".format(fft_sum, rms_sum))
    print("Proportions: fft-{}  rms-{} ".format(len(binary_labels_fft)/fft_sum, len(binary_labels_fft)/rms_sum))

def fft_animation(amplitude_fft, frequencies, framerate, binary_labels, print_labels=False, print_mean_freq=False):
    fig, ax = pl.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 0.5)  # Adjust x-axis limits as needed
    ax.set_ylim(0, 10)  # Adjust y-axis limits as needed
    ax.set_title('FFT of Amplitude Buffers')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        
        frequency = frequencies[frame]
        fft = amplitude_fft[frame]

        # debugging
        if print_labels:
            print("Part grinding:", binary_labels[frame])
        if print_mean_freq:
            print("Frequency: ", np.mean(fft[470:480]))

        line.set_data(frequency, fft)
        return line,

    # Reduce the number of frames if needed
    num_frames = len(amplitude_buffers)
    print("Num frames: ", num_frames)

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval = num_frames/framerate)

    pl.show()

if __name__ == "__main__":

    # load data
    df = pd.read_csv('vibration_data_20250313_104518.csv')
    data = df.to_numpy()
    time = data[:, 0]
    amplitude = data[:, 1]

    # convert into buffers
    buffer_size = 1000
    time_buffers = convert_to_buffers(time, buffer_size)
    amplitude_buffers = convert_to_buffers(amplitude, buffer_size)

    # assign binary classification
    rms_voltages = rms(amplitude_buffers) # obtain rms voltage
    binary_labels_rms = assign_class_rms(rms_voltages, 0.3558)

    # Plotting the RMS voltages
    pl.figure()
    pl.plot(rms_voltages, marker='o')
    pl.title('RMS Voltages for Each Amplitude Buffer')
    pl.xlabel('Buffer Index')
    pl.ylabel('RMS Voltage')
    pl.grid(True)
    pl.show()

    # Perform FFT on each buffer and create animation
    amplitude_fft, frequencies = fft_of_buffer(amplitude_buffers, buffer_size)
    binary_labels_fft = assign_class_fft(amplitude_fft, min_freq=400, max_freq=500, threshold=0.5)
    fft_animation(amplitude_fft, frequencies, 60, binary_labels_fft, print_labels=True, print_mean_freq=True)

    count_stats(binary_labels_fft, binary_labels_rms)
    

    


    

    


    


    

