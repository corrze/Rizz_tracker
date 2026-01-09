# Rizz Tracker 👔✨

A computer vision project that tracks people's movement and fashion using your webcam, displaying a dynamic "rizz" meter above their head that changes based on posture, movement patterns, and clothing detection.

## Features

- 🎥 Real-time webcam tracking
- 👤 Full-body pose detection using MediaPipe Tasks Pose Landmarker
- 👔 Fashion/accessory detection using YOLO
- 💇 Hair / head framing awareness (keeps your hair in frame for more rizz)
- 📊 Dynamic "rizz" score calculation, with **forgiving but pose-focused scoring**:
  - **Posture** (~40%): Tall stance, level shoulders, head clearly above shoulders
  - **Pose / Style** (~40%): Clean head tilt + visible hair/top of head, works for front or sideways poses
  - **Movement** (~10%): A bit of smooth motion helps; not the main factor
  - **Fashion** (~10%): Detected accessories (bag, hat, etc.) add extra rizz
- 💬 Live **tips box** on the left that tells you what to change to raise your rizz (toggleable)
- 🎨 Visual rizz meter above your head with color-coded feedback (red → orange → green)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   git clone https://github.com/corrze/Rizz_tracker/tree/main
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
- **'v' key**: Toggle pose visualization (skeleton + landmarks)
- **'t' key**: Toggle the tips box (advice on how to raise rizz)
- The camera feed shows a mirror effect (horizontally flipped)

### Tips for Higher Rizz Score

1. **Posture**: Stand tall with shoulders level and head clearly above your shoulders
2. **Pose / Style**: Hold a strong pose; keep your head tilt intentional, not slouched
3. **Hair Framing**: Adjust the camera so the top of your head and hair are fully in frame
4. **Movement**: Add a bit of smooth, controlled motion instead of jittery movements
5. **Fashion**: Bring visible accessories (hat, bag, etc.) into frame for extra style points

## How It Works

1. **Pose Detection**: MediaPipe Tasks Pose Landmarker detects 33+ body landmarks in real-time
2. **Head & Hair Tracking**: Uses nose/eyes to estimate the top of your head and hair position
3. **Movement Analysis**: Compares key joints (shoulders, hips, wrists, ankles) over time
4. **Posture Analysis**: Evaluates shoulder alignment and how high your head sits above your shoulders
5. **Pose / Style Analysis**: Scores head tilt, pose stability, and hair visibility (works for sideways poses)
6. **Fashion Detection**: YOLO detects person + accessories every 5th frame for performance
7. **Rizz Calculation**: Combines movement (~10%), posture (~40%), style (~40%), fashion (~10%), then slightly eases scores toward the high-mid range to be more forgiving
8. **Visual Display**: A rizz meter is drawn above your head, plus optional pose skeleton and tips box

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
