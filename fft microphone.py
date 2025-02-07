import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, help='Input device ID')
    parser.add_argument('-r', '--samplerate', type=float, help='Sampling rate')
    parser.add_argument('-n', '--downsample', type=int, default=5, help='Downsampling factor')
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
        plotdata[-shift:] = data  # Keep consistent shape
    line.set_ydata(plotdata)
    return line,

# Update FFT plot
def update_plot_fft(frame):
    global plotdata_fft
    plotdata_fft = np.abs(np.fft.fft(plotdata))[:len(plotdata)//2]  # Only take positive frequencies
    line_fft.set_ydata(plotdata_fft)
    return line_fft,

# Initialize parameters
args = parse_arguments()
q = queue.Queue()
device_info = sd.query_devices(args.device, 'input') if args.samplerate is None else None
samplerate = args.samplerate or device_info['default_samplerate']
plot_length = int(200 * samplerate / (1000 * args.downsample))
plotdata = np.zeros(plot_length)  # Ensure it's 1D
plotdata_fft = np.zeros(plot_length // 2)  # FFT has half the size (positive frequencies)

# Calculate the effective sampling rate after downsampling
effective_samplerate = samplerate / args.downsample

# Calculate corresponding frequency bins for FFT (positive frequencies only)
frequencies = np.fft.fftfreq(len(plotdata), 1/effective_samplerate)[:len(plotdata)//2]

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

ax2.set_ylim(0, 50)

# Start audio stream
stream = sd.InputStream(device=args.device, samplerate=samplerate, channels=1, callback=audio_callback)

ani1 = FuncAnimation(fig1, update_plot, interval=10, blit=True, cache_frame_data=False)
ani2 = FuncAnimation(fig2, update_plot_fft, interval=10, blit=True, cache_frame_data=False)

with stream:
    plt.show(block=True)  # Keep both windows open
