# OpenCV Blob Detection Optimization

Code from my time working with the **ATLAS collaboration at SCIPP**, focused on optimizing blob detection using **OpenCV** for sensor debris detection.

## Overview

This project demonstrates how to use **OpenCV’s blob detection methods** efficiently, with parameter optimization for identifying debris or particle-like features in sensor data.
It includes:

- An optimized Python script for blob detection
- A configuration file with adjustable parameters
- Example setup for reproducible experiments

## Repository Structure

```
.
├── LICENSE                      # MIT License
├── config_test.json             # JSON config file for parameter tuning
└── sensor_debris_detect_optimized.py  # Main optimized blob detection script
```

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- (Optional) Matplotlib for visualization

You can install dependencies with:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. **Configure the parameters**
   Modify `config_test.json` to adjust parameters for blob detection (e.g., threshold values, filtering criteria, area limits).

2. **Run the script**

   ```bash
   python sensor_debris_detect_optimized.py --config config_test.json
   ```

   This will:
   - Load the configured values
   - Run the optimized blob detection
   - Display or save output showing detected features

3. **Adjust & Optimize**
   Run experiments with different JSON configs to compare performance.

## Example Config (`config_test.json`)

```json
{
  "minThreshold": 50,
  "maxThreshold": 220,
  "filterByArea": true,
  "minArea": 30,
  "maxArea": 500
}
```
