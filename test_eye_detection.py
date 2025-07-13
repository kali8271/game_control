import cv2
import numpy as np
from collections import deque

class EyeDetector:
    """Eye tracking using OpenCV for game control"""
    def __init__(self):
        # Load pre-trained eye cascade classifier
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Eye tracking parameters
        self.eye_history = deque(maxlen=5)
        self.frame_skip = 2
        self.frame_count = 0
        
        # Eye control states
        self.eye_detected = False
        self.eye_position = "center"
        self.pupil_detected = False
        self.blink_detected = False
        
        # Calibration values
        self.calibration_done = False
        self.left_threshold = 0.4
        self.right_threshold = 0.6
        
    def detect_eyes(self, frame):
        """Detect eyes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Detect eyes in the face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
        
        if len(eyes) < 2:
            return None, None
        
        # Sort eyes by x position (left to right)
        eyes = sorted(eyes, key=lambda x: x[0])
        
        return eyes, (x, y, w, h)
    
    def detect_pupil(self, eye_roi):
        """Detect pupil in eye region"""
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to find dark regions (pupil)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest dark region (pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 50:  # Too small
            return None
        
        # Get pupil center
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def get_eye_position(self, eyes, face_rect, frame):
        """Determine eye gaze direction"""
        if len(eyes) < 2:
            return "center"
        
        # Get the left eye (first in sorted list)
        left_eye = eyes[0]
        x, y, w, h = left_eye
        
        # Get face coordinates
        fx, fy, fw, fh = face_rect
        
        # Create eye ROI from the original frame
        eye_roi = frame[fy + y:fy + y + h, fx + x:fx + x + w]
        pupil = self.detect_pupil(eye_roi)
        
        if pupil is None:
            return "center"
        
        pupil_x, pupil_y = pupil
        
        # Normalize pupil position relative to eye width
        relative_x = pupil_x / w
        
        # Determine gaze direction
        if relative_x < self.left_threshold:
            return "left"
        elif relative_x > self.right_threshold:
            return "right"
        else:
            return "center"
    
    def detect_blink(self, eyes):
        """Detect if eyes are closed (blink)"""
        if len(eyes) < 2:
            return True  # Eyes not detected = blink
        
        # Check if eye regions are too small (closed eyes)
        total_eye_area = sum(w * h for x, y, w, h in eyes[:2])
        if total_eye_area < 2000:  # Threshold for closed eyes
            return True
        
        return False
    
    def process_frame(self, frame):
        """Process frame for eye tracking"""
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return self.get_smoothed_results()
        
        frame = cv2.flip(frame, 1)
        eyes, face_rect = self.detect_eyes(frame)
        
        # Update eye state
        self.eye_detected = eyes is not None and len(eyes) >= 2
        self.blink_detected = self.detect_blink(eyes) if eyes is not None else True
        
        if self.eye_detected and not self.blink_detected:
            self.eye_position = self.get_eye_position(eyes, face_rect, frame)
            self.pupil_detected = True
        else:
            self.eye_position = "center"
            self.pupil_detected = False
        
        self.eye_history.append(self.eye_position)
        
        # Draw eye detection
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            
            if eyes is not None:
                for i, (x, y, w, h) in enumerate(eyes[:2]):  # Draw first 2 eyes
                    eye_x, eye_y = fx + x, fy + y
                    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + w, eye_y + h), (0, 255, 0), 2)
                    
                    # Draw eye number
                    cv2.putText(frame, f"Eye {i+1}", (eye_x, eye_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Detect and draw pupil
                    eye_roi = frame[eye_y:eye_y+h, eye_x:eye_x+w]
                    pupil = self.detect_pupil(eye_roi)
                    if pupil:
                        px, py = pupil
                        cv2.circle(frame, (eye_x + px, eye_y + py), 3, (0, 0, 255), -1)
        
        # Display eye info
        cv2.putText(frame, f"Eye: {self.eye_position.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {'YES' if self.eye_detected else 'NO'}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink: {'YES' if self.blink_detected else 'NO'}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pupil: {'YES' if self.pupil_detected else 'NO'}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.get_smoothed_results()
    
    def get_smoothed_results(self):
        """Get smoothed eye tracking results"""
        if not self.eye_history:
            return "center", False, False
        
        # Get most common position
        positions = list(self.eye_history)
        position = max(set(positions), key=positions.count)
        
        return position, self.eye_detected, self.blink_detected

def main():
    print("Eye Detection Test")
    print("=" * 20)
    print("This will test eye detection functionality.")
    print("Make sure your face is well-lit and visible to the camera.")
    print("Press 'q' to quit.")
    print()
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Starting eye detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Process frame
        eye_position, eye_detected, blink_detected = detector.process_frame(frame)
        
        # Print status
        print(f"\rEye: {eye_position.upper()} | Detected: {'YES' if eye_detected else 'NO'} | Blink: {'YES' if blink_detected else 'NO'}", end="")
        
        # Show frame
        cv2.imshow("Eye Detection Test", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nEye detection test completed.")

if __name__ == "__main__":
    main() 