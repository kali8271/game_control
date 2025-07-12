# 🎮 Gesture-Controlled Games Collection

A collection of Python games controlled by hand gestures using computer vision. Features multiple detection methods and game modes with advanced UI and scoring systems.

## 📋 Table of Contents

- [Features](#-features)
- [Games Included](#-games-included)
- [Installation](#-installation)
- [How to Play](#-how-to-play)
- [Game Controls](#-game-controls)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 **Multiple Detection Methods**
- **OpenCV Detection**: Works on any Python setup
- **MediaPipe Detection**: Advanced hand tracking (if available)
- **Palm Detection**: Simplified palm-based controls
- **Automatic Fallback**: Smart detection of available libraries

### 🎮 **Game Features**
- **Power-up System**: Double jump, speed boost, shield, time slow
- **Particle Effects**: Visual feedback for actions
- **Multiple Obstacles**: Normal, flying, and ground obstacles
- **Scoring System**: Milestone-based scoring with high score tracking
- **Advanced UI**: Game over screens, pause menus, main menu

### 🎨 **Visual Enhancements**
- **Real-time Hand Tracking**: Visual feedback in camera window
- **Particle Effects**: Explosions and visual feedback
- **Animated Power-ups**: Floating power-ups with symbols
- **Status Indicators**: Real-time power-up and game status

## 🎲 Games Included

### 1. **Unified Game** (`game_unified.py`)
- **Main game** with menu system
- **Choose detection method** (MediaPipe or OpenCV)
- **Full feature set** with all power-ups and effects
- **Professional UI** with pause and game over screens

### 2. **Palm-Controlled Game** (`game_palm_controlled.py`)
- **Simplified controls** using palm position and size
- **Fixed scoring system** (milestone-based)
- **Intuitive gameplay** - just move your palm around
- **Perfect for beginners**

### 3. **Enhanced Game** (`game_enhanced.py`)
- **Advanced gesture recognition** (fist, peace, open palm)
- **Enhanced particle effects** and visual feedback
- **Multiple game modes** and features

### 4. **Optimized Game** (`game_optimized.py`)
- **Performance optimized** with frame skipping
- **Better hand detection** with multiple skin ranges
- **Improved accuracy** and responsiveness

### 5. **Alternative Game** (`game_alternative.py`)
- **OpenCV-only version** without MediaPipe dependency
- **Basic gesture controls** with finger counting
- **Simple and reliable**

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Good lighting for hand detection

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd game_control
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Game
```bash
# Main unified game (recommended)
python game_unified.py

# Palm-controlled game
python game_palm_controlled.py

# Other games
python game_enhanced.py
python game_optimized.py
python game_alternative.py
```

## 🎮 How to Play

### **Starting the Game**
1. Run the game file
2. Navigate the menu with arrow keys
3. Select detection method (if applicable)
4. Position yourself in front of the camera
5. Ensure good lighting for hand detection

### **Basic Controls**
- **Move your hand left/right** to control character movement
- **Show fingers or bring hand closer** to jump
- **Use gestures** for special actions (depending on game)

### **Game Objectives**
- **Avoid obstacles** (red rectangles)
- **Collect power-ups** (colored squares)
- **Achieve high scores** by surviving longer
- **Beat your previous high score**

## 🎯 Game Controls

### **Finger-Based Controls** (Unified/Enhanced Games)
- **1 finger** = Move right
- **2 fingers** = Move left
- **3+ fingers** = Jump
- **Fist gesture** = Double jump (if available)
- **Open palm** = Special move

### **Palm-Based Controls** (Palm-Controlled Game)
- **Palm on left side** = Move left
- **Palm on right side** = Move right
- **Palm in center + large** = Jump
- **No finger counting needed**

### **Keyboard Controls**
- **P** = Pause game
- **Q** = Quit game
- **Arrow Keys** = Navigate menus
- **Enter** = Select menu option
- **Escape** = Back/Resume

## 🎁 Power-ups

### **🟣 Double Jump** (Magenta)
- **Effect**: Allows double jumping
- **Duration**: 10 seconds
- **Visual**: Magenta square with circle

### **🟡 Speed Boost** (Yellow)
- **Effect**: Increases movement and game speed
- **Duration**: 5 seconds
- **Visual**: Yellow square with arrow

### **🔵 Shield** (Cyan)
- **Effect**: Protects from one obstacle hit
- **Duration**: 8 seconds
- **Visual**: Cyan square with shield effect

### **🟣 Time Slow** (Purple)
- **Effect**: Slows down game time
- **Duration**: 3 seconds
- **Visual**: Purple square

## 📊 Scoring System

### **Milestone Scoring**
- **+50 points** every 5 obstacles passed
- **+25 points** for each power-up collected
- **High score tracking** with persistent storage

### **Score Display**
- **Current score** shown in real-time
- **High score** displayed when available
- **Milestone notifications** in console

## 🔧 Technical Details

### **Detection Methods**

#### **OpenCV Detection**
- **Skin color segmentation** with multiple HSV ranges
- **Contour analysis** for hand detection
- **Convexity defects** for finger counting
- **Performance optimized** with frame skipping

#### **MediaPipe Detection** (if available)
- **21 hand landmarks** for precise tracking
- **Real-time hand pose** estimation
- **Advanced gesture recognition**
- **Higher accuracy** but requires MediaPipe library

#### **Palm Detection**
- **Simplified hand tracking** using contour area
- **Position-based controls** (left/center/right)
- **Size-based actions** (jump when palm is large)
- **More reliable** in varying lighting conditions

### **Performance Optimizations**
- **Frame skipping** for better FPS
- **Efficient contour processing**
- **Smoothing algorithms** for stable detection
- **Adaptive thresholds** based on lighting

### **File Structure**
```
game_control/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── game_unified.py          # Main unified game
├── game_palm_controlled.py  # Palm-controlled game
├── game_enhanced.py         # Enhanced features game
├── game_optimized.py        # Performance optimized game
├── game_alternative.py      # OpenCV-only game
└── high_score.txt          # High score storage (auto-generated)
```

## 🛠️ Troubleshooting

### **Common Issues**

#### **Camera Not Working**
```
[ WARN:0@1.470] global cap_msmf.cpp:476 videoio(MSMF): OnReadSample() is called with error status
```
- **Solution**: This is normal on Windows, doesn't affect gameplay
- **Alternative**: Try different camera index in code

#### **Hand Detection Issues**
- **Problem**: Hand not detected properly
- **Solutions**:
  - Ensure good lighting
  - Position hand clearly in camera view
  - Try different detection method
  - Adjust skin color ranges if needed

#### **Performance Issues**
- **Problem**: Low FPS or laggy controls
- **Solutions**:
  - Use optimized game version
  - Reduce camera resolution
  - Close other applications
  - Use palm-controlled game for better performance

#### **MediaPipe Not Available**
```
MediaPipe not available - Using OpenCV hand detection
```
- **Solution**: This is normal, game will use OpenCV detection
- **To install MediaPipe**: `pip install mediapipe` (may not work on Python 3.13)

### **System Requirements**
- **OS**: Windows, macOS, Linux
- **Python**: 3.8 - 3.11 (MediaPipe), 3.8+ (OpenCV)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: Any webcam with 640x480 or higher resolution

## 🎯 Tips for Best Experience

### **Lighting**
- **Good lighting** is crucial for hand detection
- **Avoid backlighting** (bright windows behind you)
- **Use consistent lighting** throughout gameplay

### **Hand Positioning**
- **Keep hand clearly visible** in camera frame
- **Avoid rapid movements** for better detection
- **Use consistent gestures** for reliable controls

### **Performance**
- **Close unnecessary applications** for better FPS
- **Use palm-controlled game** for best performance
- **Position camera at eye level** for optimal detection

### **Game Strategy**
- **Collect power-ups** for higher scores
- **Use shield strategically** for difficult sections
- **Time your jumps** to avoid obstacles
- **Practice gesture recognition** for better control

## 🤝 Contributing

### **Adding New Features**
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

### **Suggested Improvements**
- **New gesture types** and controls
- **Additional power-ups** and effects
- **Multiplayer support**
- **Different game modes**
- **Mobile app version**

### **Bug Reports**
- **Describe the issue** clearly
- **Include system information** (OS, Python version)
- **Provide error messages** if any
- **Steps to reproduce** the problem

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **MediaPipe** for advanced hand tracking
- **Pygame** for game development framework
- **Python community** for excellent libraries

---

**Enjoy playing! 🎮✨**

For questions or support, please open an issue on the repository. 