# Face Mesh Green Screen Effect

A real-time video processing application that combines MediaPipe face mesh tracking with custom segmentation effects. Creates a stylized visual effect with person detection, brightness-based masking, and facial landmark overlays.

## Features

- **Real-time person segmentation** using MediaPipe's selfie segmenter
- **Brightness-based masking** - Separates bright and dark areas of the person
  - Bright areas: Custom purple/pink overlay
  - Dark areas: Darkened original RGB
  - Background: Green screen effect
- **Face mesh tracking** with customizable landmarks
  - Eyes outline and irises
  - Eyebrows
  - Adjustable colors, thickness, and styles
- **Performance optimized** with pre-allocated arrays and vectorized operations
- **Adjustable parameters** for darkness threshold, colors, and brightness levels

## Requirements
```
opencv-python
numpy
mediapipe
```

## Setup

1. Install dependencies:
```bash
pip install opencv-python numpy mediapipe
```

2. Download the MediaPipe selfie segmentation model:
   - Place `selfie_segmenter.tflite` in the same directory as the script
   - Download from: [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#models)

3. Run the script:
```bash
python Testing2.py
```

## Usage

- The application will open your default webcam
- Press **ESC** to exit
- The video feed is horizontally flipped for a mirror effect

## Customization

Adjust these constants at the top of the script:

- `BG_COLOR` - Background color (default: green)
- `ME_COLOR` - Color for bright person areas (default: purple/pink)
- `DARKNESS_THRESHOLD` - Threshold for separating bright/dark areas (default: 25)
- `BRIGHTNESS_FACTOR` - Grayscale brightness multiplier (default: 0.7)
- `BRIGHTNESS_FACTOR2` - RGB darkness level (default: 0.6)

Customize face mesh drawing styles:
- `eyebrow_style` - Color, thickness, and circle radius for eyebrows
- `eye_style` - Color, thickness, and circle radius for eyes/irises

## How It Works

1. **Segmentation**: Uses MediaPipe to separate person from background
2. **Brightness Analysis**: Analyzes grayscale brightness to create three regions:
   - Background → Green
   - Bright person → Custom color overlay
   - Dark person → Darkened original video
3. **Face Tracking**: Detects and draws facial landmarks on top of the processed image
4. **Display**: Outputs the combined effect in real-time

## Performance

Optimized for real-time processing at 640x480 resolution with:
- Pre-allocated array buffers
- Vectorized NumPy operations
- Minimized object creation in the main loop
- Single face detection mode

## License

Apache

## Acknowledgments

- Built with [MediaPipe](https://mediapipe.dev/) by Google
- Uses OpenCV for video processing
