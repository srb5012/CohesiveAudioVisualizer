# CohesiveAudioVisualizer
This program creates a real-time audio visualizer by leveraging several key Python libraries and signal processing techniques. It uses **PyAudio** to capture live microphone input as a stream of raw data. The core of the analysis is handled by **NumPy**, which performs a Fast Fourier Transform (FFT) to convert the time-domain audio signal into its frequency components. To achieve a precise and stable output, this process is enhanced with a Hann window to reduce spectral leakage, quadratic interpolation to pinpoint the true frequency peak, and an exponential moving average combined with a hysteresis threshold to prevent visual jitter. The resulting dominant frequency is then mapped to a specific color hue using Python's built-in **colorsys** library. The entire visual interface, including the live graphs and color display, is built and continuously updated using **Matplotlib's** FuncAnimation, creating a cohesive and dynamic visualization.

## Features
- **Live Audio Input**: Captures audio directly from a microphone.
- **Real-Time Analysis**: Performs a Fast Fourier Transform on the incoming audio signal.
- **Frequency-to-Color Mapping**: Converts the dominant frequency into a corresponding color on the visible spectrum.
- **Interactive GUI**: Displays the raw audio waveform, the frequency spectrum, and the final color in three distinct, live-updating panels.
- **Advanced Stability**: Implements smoothing and hysteresis logic to produce a stable, non-flickering color output.
- **Cohesive Design**: The colors of the graph lines dynamically update to match the detected frequency's color.

## Configuration & Tuning
The primary parameters for tuning the visualizer's behavior are located at the top of the script under the `--- Configuration Constants ---` section.

- `CHUNK`: Controls the trade-off between frequency precision and latency. Larger values are more precise but have a higher latency.
- `FFT_MAX_HZ`: Sets the upper limit for frequency analysis, useful for ignoring high-pitched noise.
- `ALPHA`: The smoothing factor. A smaller value (e.g., 0.1) results in very smooth, slow color changes. A larger value (e.g., 0.5) is more responsive but can be jittery.
- `CHANGE_THRESHOLD_HZ`: The range set that prevents the display from changing until the smoothed frequency has moved significantly away from the currently displayed one for color changes.
- `AMPLITUDE_THRESHOLD`: The noise gate. Increase this if silence or background noise is triggering colors. Decrease it if the visualizer isn't picking up quiet sounds.
