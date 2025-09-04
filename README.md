# 🖐️ Comprehensive Hand Gesture Controller

A powerful Python application that uses computer vision and hand tracking to control your computer through natural hand gestures. This project leverages MediaPipe for hand detection and OpenCV for camera processing to create an intuitive gesture-based interface.

## ✨ Features

### 🎯 Core Functionality
- **Mouse Control**: Move cursor, click, right-click, and drag using hand gestures
- **Scroll Control**: Vertical and horizontal scrolling with multi-finger gestures
- **Virtual Keyboard**: On-screen keyboard for text input
- **Multi-Mode Support**: Switch between different control modes (Normal, Keyboard, Media, Window, Gaming, Drawing)
- **Two-Hand Gestures**: Advanced zoom, rotate, and complex interactions
- **Precision Mode**: Fine-tuned control for detailed tasks

### 🎮 Gesture Categories

#### Basic Mouse Actions
- **Index Finger**: Move cursor
- **Index + Middle (close)**: Left click
- **Thumb + Index + Middle (close)**: Right click
- **3 Fingers**: Scroll (move hand up/down)
- **4 Fingers**: Double click
- **5 Fingers**: Drag mode

#### Advanced Gestures
- **L-shape (Thumb + Index)**: Screenshot
- **Peace Sign (spread fingers)**: Lock screen
- **Thumb + Middle + Ring**: Cycle through modes
- **Long Press Gestures**: Toggle precision/drawing modes

#### Specialized Modes
- **Media Control**: Volume, play/pause, track navigation
- **Window Management**: Maximize, minimize, snap windows
- **Browser Control**: Back, forward, new tab, refresh
- **Gaming Controls**: WASD movement, space bar
- **Drawing Mode**: Digital drawing with finger tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Webcam or camera device
- Windows, macOS, or Linux

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd PE\ 11
   ```

2. **Install required packages**
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

3. **Run the application**
   ```bash
   python code.py
   ```

### First Time Setup

1. **Camera Setup**
   - Ensure your camera is connected and working
   - Position yourself 1-2 feet from the camera
   - Ensure good lighting for optimal hand tracking

2. **Calibration**
   - Press 'c' in the application to enter calibration mode
   - Follow the on-screen instructions

3. **Gesture Practice**
   - Press 'h' to toggle the help overlay
   - Practice basic gestures slowly at first
   - Use the tutorial mode for guided learning

## 🎮 Controls & Gestures

### Keyboard Shortcuts
- **'q'**: Quit application
- **'h'**: Toggle help overlay
- **'c'**: Enter calibration mode
- **'s'**: Save gesture profile
- **'r'**: Reset all modes

### Gesture Reference

| Gesture | Action | Description |
|---------|--------|-------------|
| 👆 Index only | Move cursor | Point and move to control mouse |
| 👆✌️ Index + Middle | Left click | Bring fingers together to click |
| 👍👆✌️ Thumb + Index + Middle | Right click | Three fingers together |
| ✌️✌️✌️ Three fingers | Scroll | Move hand up/down to scroll |
| 🖐️ Five fingers | Drag | Hold and move to drag objects |
| 👍👆 L-shape | Screenshot | Thumb and index at 90° angle |
| ✌️ Peace sign | Lock screen | Spread index and middle fingers |

### Mode Switching
Use **Thumb + Middle + Ring** to cycle through modes:
1. **NORMAL**: Basic mouse and scroll control
2. **KEYBOARD**: Virtual keyboard and text editing
3. **MEDIA**: Music/video playback control
4. **WINDOW**: Window management and app switching
5. **GAMING**: WASD movement and game controls
6. **DRAWING**: Drawing and annotation tools

## 🔧 Configuration

### Customization Options
- **Sensitivity**: Adjust `smoothening` parameter for cursor movement speed
- **Click Threshold**: Modify `click_threshold` for click detection sensitivity
- **Gesture Cooldown**: Change `gesture_cooldown` to prevent accidental triggers
- **Precision Mode**: Long press index finger to toggle fine control

### Profile Management
- Press 's' to save current settings
- Settings are automatically loaded on startup
- Profile saved as `gesture_profile.json`

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Try different camera index in code
- Ensure camera is not being used by another application

**Poor gesture recognition:**
- Improve lighting conditions
- Use contrasting background
- Keep hand 1-2 feet from camera
- Ensure hand is fully visible in frame

**Performance issues:**
- Close other applications using camera
- Reduce camera resolution in code
- Check system resources

**Gestures not working:**
- Run calibration mode ('c')
- Check if hand is properly detected (landmarks visible)
- Try different gesture combinations
- Reset modes ('r')

### System Requirements
- **Minimum**: 4GB RAM, dual-core processor
- **Recommended**: 8GB RAM, quad-core processor
- **Camera**: 720p or higher resolution
- **OS**: Windows 10+, macOS 10.14+, or Linux

## 📁 Project Structure

```
PE 11/
├── code.py              # Main application file
├── README.md            # This documentation
└── gesture_profile.json # Saved user settings (auto-generated)
```

## 🎓 Learning Resources

### Getting Started
1. **Basic Movement**: Start with index finger cursor control
2. **Clicking**: Practice bringing index and middle fingers together
3. **Scrolling**: Use three fingers and move hand vertically
4. **Mode Switching**: Learn the thumb + middle + ring gesture

### Advanced Usage
- **Precision Mode**: Enable for detailed work like photo editing
- **Drawing Mode**: Use for digital annotation and sketching
- **Two-Hand Gestures**: Master zoom and rotate controls
- **Custom Shortcuts**: Modify the code for application-specific gestures

## 🤝 Contributing

This project is open for contributions! Areas for improvement:
- Additional gesture recognition
- Better UI/UX design
- Performance optimizations
- Cross-platform compatibility
- Accessibility features

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: Google's framework for hand tracking
- **OpenCV**: Computer vision library
- **PyAutoGUI**: Cross-platform GUI automation

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the gesture reference guide
3. Try the calibration mode
4. Check system requirements

---

**Happy Gesturing! 🖐️✨**

*Transform your computer interaction with the power of hand gestures!*
