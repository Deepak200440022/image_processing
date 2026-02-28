# Real-Time Video Processing and Motion Detection

## Overview
This project captures video from a webcam and allows real-time image processing through user key inputs. It also includes motion detection functionalities that can be triggered interactively.

## Features
- **Real-time Video Processing**: Apply transformations to the live video feed using keyboard controls.
- **Motion Detection**: Detects movement in the video using different methods.
- **Frame Differencing with Grayscale**: Detects motion by analyzing frame differences.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy
```

## Usage
Run the following command to start the interactive video processing:
```bash
python main.py
```

## Key Controls
| Key | Action |
|----|---------|
| `ESC` | Exit the program |
| `0` | Reset processing to original frame |
| `1-3,5-8,r,t,y,f,c,s,b,n,g,h,e` | Apply various transformations |
| `4` | Activate motion detection |
| `9` | Activate frame difference motion detection (grayscale) |

## Functions
### `process_video_interactive()`
Captures video from the webcam and applies transformations or motion detection based on user input.

### `apply_processing(frame, key)`
Applies the selected transformation to the input frame based on the key pressed.

### `motion_detection.motion_detection(cap)`
Performs motion detection using frame processing.

### `motion_detection.motion_detection_frame_difference(cap, grayScale=True)`
Performs motion detection using frame difference, with an optional grayscale mode.

## Notes
- Ensure your webcam is properly connected.
- Close OpenCV windows before switching between modes.
- Modify `apply_processing()` to include custom transformations.

## License
This project is open-source and free to use.
# iamge_preproecessing
