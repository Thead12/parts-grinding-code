import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
DOWNSAMPLE = 5
MIN_FREQ = 1000
MAX_FREQ = 1500
AMPLITUDE_THRESHOLD = 0.5
TIME_THRESHOLD = 100

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, help='Input device ID')
    parser.add_argument('-r', '--samplerate', type=float, help='Sampling rate')
    parser.add_argument('-n', '--downsample', type=int, default=DOWNSAMPLE, help='Downsampling factor')
    return parser.parse_args()

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, 0].squeeze())  # Convert (N,1) → (N,)

# Update time-domain plot
def update_plot(frame):
    global plotdata
    while not q.empty():
        data = q.get()
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:] = data
    line.set_ydata(plotdata)
    return line,

# Update FFT plot
def update_plot_fft(frame):
    global plotdata_fft
    plotdata_fft = np.abs(np.fft.fft(plotdata))[:len(plotdata)//2]  # Only take positive frequencies
    line_fft.set_ydata(plotdata_fft)
    check_part(plotdata_fft, min_index, max_index, AMPLITUDE_THRESHOLD, TIME_THRESHOLD)
    return line_fft,

# Counting algorithm
def check_part(fft_data, min_index, max_index, amplitude_threshold, time_threshold):
    global part_count, rolling_counter, duplicate_check
    target_avg = np.mean(fft_data[min_index:max_index])

    if target_avg >= amplitude_threshold:
        if not duplicate_check:  # Only increment if not already counted
            rolling_counter += 1
            if rolling_counter >= time_threshold:
                part_count += 1
                rolling_counter = 0
                duplicate_check = True
                print(part_count)
    else:
        rolling_counter = 0
        duplicate_check = False # Reset flag while below threshold

# Initialize parameters
args = parse_arguments()
q = queue.Queue()
device_info = sd.query_devices(args.device, 'input') if args.samplerate is None else None
samplerate = args.samplerate or device_info['default_samplerate']
plot_length = int(200 * samplerate / (1000 * args.downsample))
plotdata = np.zeros(plot_length)  # Ensure it's 1D
plotdata_fft = np.zeros(plot_length // 2)  # FFT has half the size (positive frequencies)

part_count = 0
rolling_counter = 0
duplicate_check = False


# Calculate the effective sampling rate after downsampling
effective_samplerate = samplerate / args.downsample
print("Effective samplerate:", effective_samplerate)

convert_to_index = len(plotdata_fft) / (effective_samplerate / 2)
min_index = round(MIN_FREQ * convert_to_index)
max_index = round(MAX_FREQ * convert_to_index)
print("min and max indices:", (min_index, max_index))


# Calculate corresponding frequency bins for FFT (positive frequencies only)
frequencies = np.fft.fftfreq(len(plotdata), 1/effective_samplerate)[:len(plotdata)//2]
print("real frequencies:", (frequencies[min_index], frequencies[max_index]))

# First window (Time-domain plot)
fig1 = plt.figure(num=1)
ax1 = fig1.add_subplot(111)
line, = ax1.plot(plotdata)
ax1.set_ylim(-1, 1)
ax1.set_title("Live Audio Signal")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")

# Second window (FFT plot)
fig2 = plt.figure(num=2)
ax2 = fig2.add_subplot(111)
line_fft, = ax2.plot(frequencies, plotdata_fft)  # Plot against the frequency array directly
ax2.set_xlim(0, frequencies[-1])  # Set x-axis to go from 0 to Nyquist frequency

# Custom x ticks (mapping data points to actual frequencies)
num_ticks = 6  # You can adjust this number as needed
tick_positions = np.linspace(0, len(frequencies)-1, num_ticks).astype(int)  # Select tick positions
tick_labels = [f'{frequencies[i]:.0f}' for i in tick_positions]  # Map these to frequency values

# Set x-ticks and labels
ax2.set_xticks(frequencies[tick_positions])  # Set the x-ticks to the actual frequency values

ax2.annotate(" ", xy=(MIN_FREQ, 0), xytext=(MIN_FREQ, 5), arrowprops=dict(facecolor='black', shrink=0.05))
ax2.annotate(" ", xy=(MAX_FREQ, 0), xytext=(MAX_FREQ, 5), arrowprops=dict(facecolor='black', shrink=0.05))
ax2.set_ylim(0, 50)

# Start audio stream
stream = sd.InputStream(device=args.device, samplerate=samplerate, channels=1, callback=audio_callback)

ani1 = FuncAnimation(fig1, update_plot, interval=20, blit=True, cache_frame_data=False)
ani2 = FuncAnimation(fig2, update_plot_fft, interval=20, blit=True, cache_frame_data=False)

with stream:
    plt.show(block=True)  # Keep both windows open

print("Final:", part_count)
