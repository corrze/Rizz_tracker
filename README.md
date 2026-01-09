# Rizz Tracker 👔✨

A computer vision project that tracks people's movement and fashion using your webcam, displaying a dynamic "rizz" meter above their head that changes based on posture, movement patterns, and clothing detection.

## Features

- 🎥 Real-time webcam tracking
- 👤 Human pose detection using MediaPipe
- 👔 Fashion/clothing detection using YOLO
- 📊 Dynamic "rizz" score calculation based on:
  - **Movement** (30%): Smooth, confident movements increase score
  - **Posture** (40%): Good posture (shoulders level, head up) boosts rizz
  - **Fashion** (30%): Detected clothing and accessories contribute to score
- 🎨 Visual rizz meter with color-coded feedback (red → orange → green)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/khoavo/Documents/Rizz
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: The first run will automatically download the YOLOv8n model (~6MB).

## Usage

Run the tracker:
```bash
python rizz_tracker.py
```

### Controls
- **'q' key**: Quit the application
- The camera feed will show a mirror effect (horizontally flipped)

### Tips for Higher Rizz Score

1. **Posture**: Stand up straight with shoulders level and head up
2. **Movement**: Make smooth, confident movements (not too fast, not too slow)
3. **Fashion**: Wear visible clothing/accessories that the system can detect
4. **Consistency**: Maintain good posture and movement over time

## How It Works

1. **Pose Detection**: MediaPipe Pose detects 33 body landmarks in real-time
2. **Head Tracking**: The system locates your head position using nose/eye landmarks
3. **Movement Analysis**: Tracks changes in key body points over time frames
4. **Posture Analysis**: Evaluates shoulder alignment and head position
5. **Fashion Detection**: YOLO model detects person and accessories (runs every 5th frame for performance)
6. **Rizz Calculation**: Combines all factors with weighted scoring to produce 0-100 rizz score
7. **Visual Display**: Dynamic meter above head shows current rizz level with color coding

## Customization

You can modify the `RizzTracker` class in `rizz_tracker.py` to adjust:

- **Scoring weights**: Change the percentages in `calculate_rizz_score()` method
- **Movement thresholds**: Adjust `movement_threshold` and multipliers
- **Camera index**: Change `camera_index` parameter if using a different camera
- **Model selection**: Upgrade YOLO model (e.g., `yolov8s.pt`, `yolov8m.pt`) for better detection at cost of speed

## Requirements

- Python 3.8+
- Webcam
- OpenCV 4.8+
- MediaPipe 0.10+
- Ultralytics YOLO 8.1+
- NumPy

## Performance

- Runs at ~15-30 FPS depending on hardware
- YOLO detection runs every 5th frame to maintain performance
- Optimized for real-time tracking on consumer hardware

## Troubleshooting

**Camera not opening:**
- Check camera permissions
- Try changing `camera_index` to 1 or 2
- Ensure no other application is using the camera

**Low FPS:**
- Reduce YOLO detection frequency (change `fps_counter % 5` to higher number)
- Use smaller YOLO model (currently using `yolov8n.pt` nano version)
- Reduce camera resolution in OpenCV settings

**No person detected:**
- Ensure good lighting
- Stand fully in frame
- Check camera focus

## License

This project is open source and available for personal and educational use.

## Future Enhancements

Potential improvements:
- More detailed fashion classification (specific clothing types)
- Facial expression analysis
- Gesture recognition
- Multi-person tracking
- Historical rizz trends/graphs
- Custom scoring algorithms

Enjoy tracking your rizz! 🚀
