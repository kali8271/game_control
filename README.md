# 🎮 Gesture & Eye-Controlled Games Collection

A collection of Python games controlled by hand gestures and eye movements using computer vision. Features multiple detection methods, eye tracking, and game modes with advanced UI and scoring systems.

## 📋 Table of Contents

- [Features](#-features)
- [Games Included](#-games-included)
- [Installation](#-installation)
- [How to Play](#-how-to-play)
- [Game Controls](#-game-controls)
- [Eye Control Guide](#-eye-control-guide)
- [Scoring System](#-scoring-system)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 **Multiple Detection Methods**
- **OpenCV Detection**: Works on any Python setup
- **MediaPipe Detection**: Advanced hand tracking (if available)
- **Palm Detection**: Simplified palm-based controls
- **👁️ Eye Detection**: Eye movement and blink tracking
- **Automatic Fallback**: Smart detection of available libraries

### 🎮 **Game Features**
- **Power-up System**: Double jump, speed boost, shield, time slow
- **Particle Effects**: Visual feedback for actions
- **Multiple Obstacles**: Normal, flying, and ground obstacles
- **📊 Enhanced Scoring System**: Frequent updates with multiple scoring opportunities
- **Advanced UI**: Game over screens, pause menus, main menu

### 🎨 **Visual Enhancements**
- **Real-time Hand/Eye Tracking**: Visual feedback in camera window
- **Particle Effects**: Explosions and visual feedback
- **Animated Power-ups**: Floating power-ups with symbols
- **Status Indicators**: Real-time power-up and game status

## 🎲 Games Included

### 1. **Unified Game** (`game_unified.py`)
- **Main game** with menu system
- **Choose detection method** (MediaPipe or OpenCV)
- **Full feature set** with all power-ups and effects
- **Professional UI** with pause and game over screens

### 2. **Eye/Palm-Controlled Game** (`game_palm_controlled.py`) ⭐ **NEW**
- **👁️ Eye Control**: Look left/right to move, blink to jump
- **🤚 Palm Control**: Move palm left/right to move, bring palm closer to jump
- **Choice menu** to select control method
- **All power-ups work** with both control methods
- **Perfect for accessibility** and unique gaming experience

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
- Good lighting for hand/eye detection

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

# Eye/Palm-controlled game (NEW!)
python game_palm_controlled.py

# Other games
python game_enhanced.py
python game_optimized.py
python game_alternative.py
```

## 🎮 How to Play

### **Starting the Game**
1. Run the game file
2. **For Eye/Palm Game**: Choose control method (1 for Eye, 2 for Palm)
3. Navigate the menu with arrow keys
4. Select detection method (if applicable)
5. Position yourself in front of the camera
6. Ensure good lighting for detection

### **Basic Controls**
- **Hand Control**: Move your hand left/right to control character movement
- **Eye Control**: Look left/right to control character movement
- **Jump**: Show fingers, bring hand closer, or blink (depending on control method)
- **Use gestures** for special actions (depending on game)

### **Game Objectives**
- **Avoid obstacles** (red rectangles)
- **Collect power-ups** (colored squares)
- **Achieve high scores** by surviving longer
- **Beat your previous high score**

## 🎯 Game Controls

### **👁️ Eye Controls** (NEW!)
- **👀 Look Left** = Move character left
- **👀 Look Right** = Move character right
- **👁️ Blink** = Jump
- **👀 Look Center** = No movement

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

## 👁️ Eye Control Guide

### **Setup for Eye Control**
1. **Good Lighting**: Make sure your face is well-lit and clearly visible to the camera
2. **Camera Position**: Position yourself so your face is clearly visible in the camera
3. **Distance**: Stay at a comfortable distance (about 1-2 feet from the camera)

### **Tips for Better Eye Control**
1. **Keep your head relatively still** - only move your eyes
2. **Make deliberate eye movements** - look clearly left or right
3. **Blink naturally** - don't force blinks, just blink normally
4. **Good lighting is crucial** - avoid shadows on your face
5. **Remove glasses if they cause glare** (if comfortable)

### **Testing Eye Detection**
You can test the eye detection separately by running:
```bash
python test_eye_detection.py
```

This will show you:
- Face detection (blue rectangle)
- Eye detection (green rectangles)
- Pupil tracking (red dots)
- Real-time status information

## 🎁 Power-ups

### **🟣 Double Jump** (Magenta)
- **Effect**: Allows double jumping
- **Duration**: 10 seconds
- **Visual**: Magenta square with circle
- **Eye Control**: Blink twice quickly to double jump

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

### **🎯 Comprehensive Scoring Breakdown**
- **+10 points** for every obstacle passed
- **+40 bonus points** every 5 obstacles (milestone)
- **+20 points** every 10 seconds of survival
- **+5 points** for successful landings
- **+25 points** for each power-up collected

### **📈 Score Progression Example**
```
Time 0s:   Score 0    (Game starts)
Time 2s:   Score 10   (1st obstacle passed)
Time 4s:   Score 20   (2nd obstacle passed)
Time 6s:   Score 30   (3rd obstacle passed)
Time 8s:   Score 40   (4th obstacle passed)
Time 10s:  Score 60   (5th obstacle + milestone + survival bonus)
Time 12s:  Score 70   (6th obstacle passed)
Time 20s:  Score 120  (10th obstacle + milestone + survival bonus)
```

### **🎮 Real-Time Feedback**
The game provides console feedback for all scoring events:
- `"Obstacle passed! Score: X"` - Every obstacle
- `"Milestone reached! Score: X"` - Every 5 obstacles
- `"Survival bonus! Score: X"` - Every 10 seconds
- `"Safe landing! Score: X"` - Successful jumps
- `"Power-up collected! Score: X"` - Power-up collection

### **🏆 High Score System**
- **Persistent storage** in `high_score.txt`
- **Automatic updates** when new high score is achieved
- **Display** shows current score and high score

### **🎯 Scoring Strategy**
- **Survive longer** - Time-based bonuses add up
- **Collect power-ups** - 25 points each
- **Jump safely** - Landing bonuses for good technique
- **Avoid obstacles** - Each one passed = 10 points
- **Plan milestones** - Every 5 obstacles = big bonus

## 🔧 Technical Details

### **Detection Methods**

#### **👁️ Eye Detection** (NEW!)
- **Face detection** using `haarcascade_frontalface_default.xml`
- **Eye detection** using `haarcascade_eye.xml`
- **Pupil tracking** using threshold-based contour analysis
- **Blink detection** using eye area analysis
- **Gaze direction** based on pupil position within eye

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
├── game_palm_controlled.py  # Eye/Palm-controlled game ⭐
├── test_eye_detection.py    # Eye detection test tool
├── game_enhanced.py         # Enhanced features game
├── game_optimized.py        # Performance optimized game
├── game_alternative.py      # OpenCV-only game
├── SCORING_SYSTEM.md        # Detailed scoring documentation
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

#### **👁️ Eye Detection Issues**
- **Problem**: Eyes not detected properly
- **Solutions**:
  - Ensure good lighting on your face
  - Remove obstructions (hair, glasses, shadows)
  - Adjust distance from camera
  - Check camera is working properly
  - Test with `python test_eye_detection.py`

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
  - **Eye detection** may be slightly more CPU-intensive

#### **Scoring Issues**
- **Problem**: Score not updating frequently
- **Solution**: ✅ **FIXED** - The scoring system has been improved to provide frequent updates
  - **Before**: Score only increased every 5 obstacles
  - **After**: Score increases with every obstacle + multiple bonus opportunities
  - **Console feedback** shows all scoring events in real-time

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
- **Good lighting** is crucial for both hand and eye detection
- **Avoid backlighting** (bright windows behind you)
- **Use consistent lighting** throughout gameplay
- **For eye control**: Ensure face is well-lit without shadows

### **Positioning**
- **Keep hand/face clearly visible** in camera frame
- **Avoid rapid movements** for better detection
- **Use consistent gestures** for reliable controls
- **For eye control**: Keep head relatively still, move only eyes

### **Performance**
- **Close unnecessary applications** for better FPS
- **Use palm-controlled game** for best performance
- **Position camera at eye level** for optimal detection
- **Eye detection** may require more processing power

### **Game Strategy**
- **Collect power-ups** for higher scores
- **Use shield strategically** for difficult sections
- **Time your jumps** to avoid obstacles
- **Practice gesture/eye recognition** for better control

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
- **Enhanced eye tracking** with calibration
- **Voice control integration**

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

**Enjoy playing with your hands AND eyes! 🎮👁️✨**

For questions or support, please open an issue on the repository. 