# Author: Shawn Balgobind
# Title: Cohesive Audio Visualizer
# Description: A program that visualizes microphone audio by performing an Fast
#              Fourier Transform and mapping the dominant frequency to a color.

# --- Imports ---
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colorsys

# --- Configuration Constants ---

# Audio processing parameters
CHUNK = 1024 * 4               # Number of audio samples per frame. (Larger -> more precision, more latency).
FORMAT = pyaudio.paInt16       # Audio format: 16-bit signed integers.
CHANNELS = 1                   # Number of audio channels
RATE = 44100                   # Sample rate in Hz.
freqResolution = RATE / CHUNK # The frequency range of each FFT bin.

# Analysis and display parameters
FFT_MAX_HZ = 2000              # The highest frequency to consider in the analysis.
MIN_FREQ_HZ = 60               # The lowest frequency to map to the color spectrum.
MAX_FREQ_HZ = 1500             # The highest frequency to map to the color spectrum.

# Stability and smoothing parameters
ALPHA = 0.15                   # Smoothing factor for frequency detection (smaller is more smooth).
CHANGE_THRESHOLD_HZ = 5        # How much frequency must change to trigger a color update.
AMPLITUDE_THRESHOLD = 30       # The minimum FFT magnitude to be considered a signal.

# State variables (values are retained between frames)
displayedFreqHz = 0.0        # The frequency currently being displayed.
smoothedTargetFreq = 0.0     # The heavily smoothed potential frequency.


# --- Audio Stream Setup ---
pyaudioInstance = pyaudio.PyAudio()

# Open the audio input stream.
audioStream = pyaudioInstance.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# --- GUI Setup ---

# Create a figure and three vertically stacked subplots.
figure, (axis1, axis2, axis3) = plt.subplots(3, 1, figsize=(8, 10))
figure.suptitle('Audio To Color', fontsize=16, color='white')
figure.patch.set_facecolor('0.1')

# Plot 1: Raw Audio Waveform
axis1.set_title("Raw Audio Wave", color='white')
axis1.set_ylim(-8000, 8000)
axis1.set_xticks([])
axis1.set_yticks([])
axis1.set_facecolor('0.25')
lineWaveform, = axis1.plot(np.arange(0, CHUNK), np.zeros(CHUNK), color='k', lw=1.5)

# Plot 2: Fast Fourier Transform Frequency Spectrum
axis2.set_title("Frequency Spectrum", color='white')
axis2.set_xlabel("Frequency (Hz)", color='white')
axis2.set_xlim(0, FFT_MAX_HZ)
axis2.set_ylim(0, 1000)
axis2.set_yticks([])
axis2.set_facecolor('0.25')
axis2.tick_params(axis='x', colors='white')
xFft = np.linspace(0, RATE / 2, CHUNK // 2)
lineFft, = axis2.plot(xFft, np.zeros(CHUNK // 2), color='k', lw=2)

# Plot 3: Color Display
axis3.set_title("Stable Frequency", color='white') # This title is updated dynamically.
axis3.set_xticks([])
axis3.set_yticks([])
axis3.set_facecolor('k')

# Set all plot borders (spines) to white.
for ax in [axis1, axis2, axis3]:
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

# --- Helper Functions ---

def frequencyToColor(freq):
    """Converts a frequency in Hz to an RGB color tuple."""
    if freq is None or freq < MIN_FREQ_HZ:
        return (0, 0, 0) # Return black for silence or low frequencies.
    
    # Normalize the frequency to a 0-1 range using a logarithmic scale.
    logMin = np.log10(MIN_FREQ_HZ)
    logMax = np.log10(MAX_FREQ_HZ)
    logFreq = np.log10(max(MIN_FREQ_HZ, min(freq, MAX_FREQ_HZ)))
    hue = (logFreq - logMin) / (logMax - logMin)
    
    # Map the hue value to a blue-to-red spectrum (0.66 -> 0.0).
    hue = 0.66 - (hue * 0.66)
    
    # Convert from HSV color.
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)

# --- Main Animation Loop ---
def updateFrame(frame):
    """Called repeatedly by FuncAnimation to process and draw each frame."""
    global smoothedTargetFreq, displayedFreqHz

    # Read audio data from the stream.
    try:
        rawData = audioStream.read(CHUNK, exception_on_overflow=False)
        audioData = np.frombuffer(rawData, dtype=np.int16)
    except IOError:
        return lineWaveform, lineFft # Skip frame on read error.

    # Perform FFT on the audio data.
    window = np.hanning(len(audioData)) # Apply a Hann window.
    dataWindowed = audioData * window
    fftData = np.fft.fft(dataWindowed)
    fftMagnitude = np.abs(fftData)[:CHUNK // 2] * 2 / CHUNK # Calculate magnitude and normalize.

    # Find the dominant frequency with stability logic.
    fftMaxIndex = int(FFT_MAX_HZ / freqResolution)
    fftMagnitudeLimited = fftMagnitude[:fftMaxIndex]
    maxMagnitude = np.max(fftMagnitudeLimited)

    # Only process if the signal is loud enough.
    if maxMagnitude > AMPLITUDE_THRESHOLD:
        peakIndex = np.argmax(fftMagnitudeLimited)
        currentFreq = 0
        
        # Interpolate for more precise peaks
        if peakIndex > 0 and peakIndex < len(fftMagnitudeLimited) - 1:
            y0, y1, y2 = fftMagnitudeLimited[peakIndex-1:peakIndex+2]
            if (y0 - 2 * y1 + y2) != 0:
                offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
                currentFreq = (peakIndex + offset) * freqResolution
        
        # Exponnentially smooth the frequency
        smoothedTargetFreq = ALPHA * currentFreq + (1 - ALPHA) * smoothedTargetFreq
        
        # Update the display on significant changes
        if abs(smoothedTargetFreq - displayedFreqHz) > CHANGE_THRESHOLD_HZ:
            displayedFreqHz = smoothedTargetFreq
    else:
        # If silent, slowly decay the frequency towards zero
        displayedFreqHz *= 0.95
        if displayedFreqHz < MIN_FREQ_HZ:
            displayedFreqHz = 0

    # Update all GUI elements with the new data
    color = frequencyToColor(displayedFreqHz)
    
    lineWaveform.set_ydata(audioData)
    lineFft.set_ydata(fftMagnitude)
    axis3.set_facecolor(color)

    # Update the title of the color panel
    if displayedFreqHz > MIN_FREQ_HZ:
        axis3.set_title(f"Stable Frequency: {displayedFreqHz:.1f} Hz", color='white')
    else:
        axis3.set_title("Silence", color='white')

    # Update the graph line colors to match the detected color.
    lineColor = color if color != (0,0,0) else (0.3, 0.3, 0.3)
    lineWaveform.set_color(lineColor)
    lineFft.set_color(lineColor)
    
    return lineWaveform, lineFft

# --- Run Application ---
# Create and start the animation, calling the update function repeatedly.
animation = FuncAnimation(figure, updateFrame, blit=False, interval=10)
plt.tight_layout(pad=3.0)
plt.show()

# --- Cleanup ---
audioStream.stop_stream()
audioStream.close()
pyaudioInstance.terminate()
