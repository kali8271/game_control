import cv2
import pygame
import sys
import numpy as np
import time
import random
import math
from collections import deque

# Try to import MediaPipe, but don't fail if it's not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe detected - Advanced hand tracking available!")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - Using OpenCV hand detection")

class Particle:
    def __init__(self, x, y, color, velocity_x=0, velocity_y=0):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.life = 30
        self.max_life = 30
    
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += 0.5
        self.life -= 1
    
    def draw(self, screen):
        if self.life > 0:
            size = int(5 * (self.life / self.max_life))
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

class PowerUp:
    def __init__(self, x, y, power_type):
        self.rect = pygame.Rect(x, y, 30, 30)
        self.power_type = power_type
        self.collected = False
        self.animation_time = 0
        
        self.colors = {
            'double_jump': (255, 0, 255),
            'speed_boost': (255, 255, 0),
            'shield': (0, 255, 255),
            'time_slow': (128, 0, 128)
        }
    
    def update(self):
        self.animation_time += 0.1
        self.rect.y += math.sin(self.animation_time) * 2
    
    def draw(self, screen):
        if not self.collected:
            color = self.colors.get(self.power_type, (255, 255, 255))
            pygame.draw.rect(screen, color, self.rect)
            
            if self.power_type == 'double_jump':
                pygame.draw.circle(screen, (255, 255, 255), self.rect.center, 8)
            elif self.power_type == 'speed_boost':
                pygame.draw.polygon(screen, (255, 255, 255), [
                    (self.rect.centerx - 8, self.rect.centery + 8),
                    (self.rect.centerx + 8, self.rect.centery),
                    (self.rect.centerx - 8, self.rect.centery - 8)
                ])

class PalmDetector:
    """Palm-based detection using OpenCV"""
    def __init__(self):
        self.skin_ranges = [
            (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 50, 50], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 30, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
        ]
        
        self.palm_history = deque(maxlen=5)
        self.frame_skip = 2
        self.frame_count = 0
        
        # Palm gesture states
        self.palm_detected = False
        self.palm_position = "center"
        self.palm_size = 0
    
    def get_skin_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.skin_ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, range_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_palm(self, frame):
        mask = self.get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "no_palm", 0
        
        # Find the largest contour (assumed to be the palm)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 5000:  # Too small to be a palm
            return None, "no_palm", 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        frame_center = frame.shape[1] // 2
        
        # Determine palm position
        margin = 100
        if center_x < frame_center - margin:
            position = "left"
        elif center_x > frame_center + margin:
            position = "right"
        else:
            position = "center"
        
        # Determine palm size (for jump detection)
        palm_size = w * h
        
        return largest_contour, position, palm_size
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return self.get_smoothed_results()
        
        frame = cv2.flip(frame, 1)
        contour, position, size = self.detect_palm(frame)
        
        # Update palm state
        self.palm_detected = contour is not None
        self.palm_position = position
        self.palm_size = size
        
        self.palm_history.append((position, size))
        
        # Draw palm detection
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw palm center
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Display palm info
        cv2.putText(frame, f"Palm: {position.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {size//1000}k", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {'YES' if self.palm_detected else 'NO'}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.get_smoothed_results()
    
    def get_smoothed_results(self):
        if not self.palm_history:
            return "center", 0, "no_palm"
        
        # Get most common position
        positions = [pos for pos, _ in self.palm_history]
        position = max(set(positions), key=positions.count)
        
        # Get average size
        sizes = [size for _, size in self.palm_history]
        avg_size = sum(sizes) // len(sizes)
        
        return position, avg_size, "palm" if self.palm_detected else "no_palm"

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

class GameOverScreen:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        
        self.font = pygame.font.Font(None, 64)
        self.large_font = pygame.font.Font(None, 96)
        self.small_font = pygame.font.Font(None, 32)
        
        self.selected_option = 0
        self.options = ["Play Again", "Main Menu", "Exit"]
    
    def draw(self, final_score, high_score=None):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        pygame.display.get_surface().blit(overlay, (0, 0))
        
        title = self.large_font.render("GAME OVER", True, self.RED)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 150))
        pygame.display.get_surface().blit(title, title_rect)
        
        score_text = self.font.render(f"Final Score: {final_score}", True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 250))
        pygame.display.get_surface().blit(score_text, score_rect)
        
        if high_score and final_score >= high_score:
            high_score_text = self.font.render("NEW HIGH SCORE!", True, self.GREEN)
            high_score_rect = high_score_text.get_rect(center=(self.WIDTH // 2, 320))
            pygame.display.get_surface().blit(high_score_text, high_score_rect)
        elif high_score:
            high_score_text = self.small_font.render(f"High Score: {high_score}", True, self.GRAY)
            high_score_rect = high_score_text.get_rect(center=(self.WIDTH // 2, 320))
            pygame.display.get_surface().blit(high_score_text, high_score_rect)
        
        for i, option in enumerate(self.options):
            color = self.BLUE if i == self.selected_option else self.WHITE
            text = self.font.render(option, True, color)
            rect = text.get_rect(center=(self.WIDTH // 2, 400 + i * 60))
            pygame.display.get_surface().blit(text, rect)
            
            if i == self.selected_option:
                pygame.draw.rect(pygame.display.get_surface(), self.BLUE, 
                               (rect.left - 20, rect.centery - 5, 10, 10))
        
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.GRAY)
            pygame.display.get_surface().blit(text, (50, self.HEIGHT - 80 + i * 30))
        
        pygame.display.flip()
    
    def run(self, final_score, high_score=None):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "exit"
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_option = (self.selected_option - 1) % len(self.options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % len(self.options)
                    elif event.key == pygame.K_RETURN:
                        return self.options[self.selected_option].lower().replace(" ", "_")
                    elif event.key == pygame.K_ESCAPE:
                        return "main_menu"
            
            self.draw(final_score, high_score)
            pygame.time.Clock().tick(60)
        
        return "exit"

class ScoreManager:
    def __init__(self):
        self.high_score = 0
        self.load_high_score()
    
    def load_high_score(self):
        try:
            with open("high_score.txt", "r") as f:
                self.high_score = int(f.read().strip())
        except:
            self.high_score = 0
    
    def save_high_score(self, score):
        if score > self.high_score:
            self.high_score = score
            try:
                with open("high_score.txt", "w") as f:
                    f.write(str(score))
            except:
                pass
        return self.high_score
    
    def get_high_score(self):
        return self.high_score

class PalmControlledGame:
    def __init__(self, control_mode="palm"):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Eye/Palm-Controlled Game")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        
        # Game objects
        self.player = pygame.Rect(100, 500, 40, 60)
        self.obstacles = []
        self.power_ups = []
        self.particles = []
        self.score = 0
        self.game_speed = 5
        self.jumping = False
        self.double_jumping = False
        self.jump_velocity = 0
        self.gravity = 0.8
        
        # Power-up states
        self.double_jump_available = False
        self.speed_boost_active = False
        self.shield_active = False
        self.time_slow_active = False
        self.power_up_timers = {}
        
        # UI Components
        self.game_over_screen = GameOverScreen(self.WIDTH, self.HEIGHT)
        self.score_manager = ScoreManager()
        
        # Game state
        self.game_running = True
        
        # Performance
        self.clock = pygame.time.Clock()
        self.last_obstacle_time = time.time()
        self.last_power_up_time = time.time()
        self.obstacle_spawn_rate = 2.0
        self.power_up_spawn_rate = 5.0
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Control mode
        self.control_mode = control_mode
        
        # Detection systems
        if control_mode == "eye":
            self.detector = EyeDetector()
            self.palm_detector = None
        else:
            self.detector = PalmDetector()
            self.palm_detector = self.detector
        
        # Scoring system - IMPROVED
        self.obstacles_passed = 0
        self.score_update_interval = 5  # Bonus every 5 obstacles passed
        self.game_start_time = time.time()
        self.last_time_score = 0
    
    def create_particles(self, x, y, color, count=10):
        for _ in range(count):
            velocity_x = random.uniform(-3, 3)
            velocity_y = random.uniform(-5, -1)
            particle = Particle(x, y, color, velocity_x, velocity_y)
            self.particles.append(particle)
    
    def handle_jump(self):
        if self.jumping:
            self.player.y -= self.jump_velocity
            self.jump_velocity -= self.gravity
            
            if self.player.y >= 500:
                self.player.y = 500
                self.jumping = False
                self.double_jumping = False
                self.jump_velocity = 0
                
                # Small score bonus for successful landing
                if not hasattr(self, 'last_landing_score'):
                    self.last_landing_score = 0
                
                current_time = time.time()
                if current_time - self.last_landing_score > 1.0:  # Prevent spam
                    self.score += 5
                    self.last_landing_score = current_time
                    print(f"Safe landing! Score: {self.score}")
    
    def spawn_obstacles(self):
        current_time = time.time()
        if current_time - self.last_obstacle_time > self.obstacle_spawn_rate:
            obstacle_type = random.choice(['normal', 'flying', 'ground'])
            
            if obstacle_type == 'normal':
                obstacle = pygame.Rect(self.WIDTH, 500, 30, 60)
            elif obstacle_type == 'flying':
                obstacle = pygame.Rect(self.WIDTH, 400, 30, 30)
            else:
                obstacle = pygame.Rect(self.WIDTH, 520, 30, 40)
            
            self.obstacles.append({'rect': obstacle, 'type': obstacle_type})
            self.last_obstacle_time = current_time
    
    def spawn_power_ups(self):
        current_time = time.time()
        if current_time - self.last_power_up_time > self.power_up_spawn_rate:
            power_types = ['double_jump', 'speed_boost', 'shield', 'time_slow']
            power_type = random.choice(power_types)
            power_up = PowerUp(self.WIDTH, random.randint(300, 450), power_type)
            self.power_ups.append(power_up)
            self.last_power_up_time = current_time
    
    def update_obstacles(self):
        speed = self.game_speed * 2 if self.speed_boost_active else self.game_speed
        
        for obstacle_data in self.obstacles[:]:
            obstacle = obstacle_data['rect']
            obstacle.x -= speed
            
            if self.player.colliderect(obstacle):
                if self.shield_active:
                    self.shield_active = False
                    self.create_particles(obstacle.centerx, obstacle.centery, self.CYAN, 20)
                else:
                    return False
            
            if obstacle.x < -obstacle.width:
                self.obstacles.remove(obstacle_data)
                self.obstacles_passed += 1
                
                # IMPROVED SCORING: Score for every obstacle + bonus milestones
                self.score += 10  # Base score for each obstacle
                
                # Bonus milestone every 5 obstacles
                if self.obstacles_passed % self.score_update_interval == 0:
                    self.score += 40  # Additional 40 points for milestone (50 total)
                    print(f"Milestone reached! Score: {self.score}")
                else:
                    print(f"Obstacle passed! Score: {self.score}")
        
        return True
    
    def update_power_ups(self):
        for power_up in self.power_ups[:]:
            power_up.update()
            
            if self.player.colliderect(power_up.rect) and not power_up.collected:
                power_up.collected = True
                self.activate_power_up(power_up.power_type)
                self.create_particles(power_up.rect.centerx, power_up.rect.centery, 
                                   power_up.colors[power_up.power_type], 15)
                self.power_ups.remove(power_up)
                
                # Score for collecting power-up
                self.score += 25
                print(f"Power-up collected! Score: {self.score}")
            
            elif power_up.rect.x < -power_up.rect.width:
                self.power_ups.remove(power_up)
    
    def activate_power_up(self, power_type):
        if power_type == 'double_jump':
            self.double_jump_available = True
            self.power_up_timers['double_jump'] = 10.0
        elif power_type == 'speed_boost':
            self.speed_boost_active = True
            self.power_up_timers['speed_boost'] = 5.0
        elif power_type == 'shield':
            self.shield_active = True
            self.power_up_timers['shield'] = 8.0
        elif power_type == 'time_slow':
            self.time_slow_active = True
            self.power_up_timers['time_slow'] = 3.0
    
    def update_power_up_timers(self, dt):
        for power_type, timer in list(self.power_up_timers.items()):
            self.power_up_timers[power_type] -= dt
            if self.power_up_timers[power_type] <= 0:
                if power_type == 'double_jump':
                    self.double_jump_available = False
                elif power_type == 'speed_boost':
                    self.speed_boost_active = False
                elif power_type == 'shield':
                    self.shield_active = False
                elif power_type == 'time_slow':
                    self.time_slow_active = False
                del self.power_up_timers[power_type]
    
    def update_particles(self):
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)
    
    def reset_game(self):
        self.player = pygame.Rect(100, 500, 40, 60)
        self.obstacles = []
        self.power_ups = []
        self.particles = []
        self.score = 0
        self.obstacles_passed = 0
        self.jumping = False
        self.double_jumping = False
        self.jump_velocity = 0
        self.double_jump_available = False
        self.speed_boost_active = False
        self.shield_active = False
        self.time_slow_active = False
        self.power_up_timers = {}
        self.last_obstacle_time = time.time()
        self.last_power_up_time = time.time()
        self.game_running = True
    
    def draw(self, position, size, detected, blink_detected=False):
        self.screen.fill(self.WHITE)
        
        pygame.draw.rect(self.screen, self.BLACK, (0, 560, self.WIDTH, 140))
        
        for particle in self.particles:
            particle.draw(self.screen)
        
        player_color = self.GREEN
        if self.shield_active:
            player_color = self.CYAN
            pygame.draw.circle(self.screen, self.CYAN, self.player.center, 35, 3)
        
        pygame.draw.rect(self.screen, player_color, self.player)
        
        for obstacle_data in self.obstacles:
            obstacle = obstacle_data['rect']
            color = self.RED if obstacle_data['type'] == 'normal' else self.YELLOW
            pygame.draw.rect(self.screen, color, obstacle)
        
        for power_up in self.power_ups:
            power_up.draw(self.screen)
        
        # UI - Score and High Score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        high_score = self.score_manager.get_high_score()
        if high_score > 0:
            high_score_text = self.font.render(f"High Score: {high_score}", True, self.BLUE)
            self.screen.blit(high_score_text, (10, 50))
        
        # Detection info based on control mode
        if self.control_mode == "eye":
            # Eye detection info
            eye_text = self.font.render(f"Eye: {position.upper()}", True, self.BLACK)
            self.screen.blit(eye_text, (10, 90 if high_score > 0 else 50))
            
            detected_text = self.font.render(f"Detected: {'YES' if detected else 'NO'}", True, self.BLACK)
            self.screen.blit(detected_text, (10, 130 if high_score > 0 else 90))
            
            blink_text = self.font.render(f"Blink: {'YES' if blink_detected else 'NO'}", True, self.BLACK)
            self.screen.blit(blink_text, (10, 170 if high_score > 0 else 130))
        else:
            # Palm detection info
            palm_text = self.font.render(f"Palm: {position.upper()}", True, self.BLACK)
            self.screen.blit(palm_text, (10, 90 if high_score > 0 else 50))
            
            size_text = self.font.render(f"Size: {size//1000}k", True, self.BLACK)
            self.screen.blit(size_text, (10, 130 if high_score > 0 else 90))
            
            detected_text = self.font.render(f"Detected: {'YES' if detected else 'NO'}", True, self.BLACK)
            self.screen.blit(detected_text, (10, 170 if high_score > 0 else 130))
        
        # Power-up status
        y_offset = 210 if high_score > 0 else 170
        if self.double_jump_available:
            text = self.small_font.render("Double Jump Available!", True, self.MAGENTA)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        if self.speed_boost_active:
            text = self.small_font.render("Speed Boost Active!", True, self.YELLOW)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        if self.shield_active:
            text = self.small_font.render("Shield Active!", True, self.CYAN)
            self.screen.blit(text, (10, y_offset))
            y_offset += 25
        
        if self.time_slow_active:
            text = self.small_font.render("Time Slow Active!", True, self.MAGENTA)
            self.screen.blit(text, (10, y_offset))
        
        # Instructions based on control mode
        if self.control_mode == "eye":
            instructions = [
                "Eye Controls:",
                "Look left = Move Left",
                "Look right = Move Right", 
                "Blink = Jump",
                "Collect power-ups for points",
                "Press 'q' to quit"
            ]
        else:
            instructions = [
                "Palm Controls:",
                "Left side = Move Left",
                "Right side = Move Right", 
                "Center + Large = Jump",
                "Collect power-ups for points",
                "Press 'q' to quit"
            ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (self.WIDTH - 250, 10 + i * 25))
        
        pygame.display.flip()
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if self.control_mode == "eye":
            print("Eye-Controlled Game Started!")
            print("Controls: Look left/right to move, blink to jump")
        else:
            print("Palm-Controlled Game Started!")
            print("Controls: Move palm left/right to move, bring palm closer to jump")
        
        running = True
        last_time = time.time()
        
        while running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame based on control mode
            if self.control_mode == "eye":
                eye_position, eye_detected, blink_detected = self.detector.process_frame(frame)
                
                # Game logic based on eye position and blink
                if eye_position == "right":
                    speed = 12 if self.speed_boost_active else 8
                    self.player.x = min(self.WIDTH - self.player.width, self.player.x + speed)
                elif eye_position == "left":
                    speed = 12 if self.speed_boost_active else 8
                    self.player.x = max(0, self.player.x - speed)
                elif blink_detected:  # Blink = jump
                    if not self.jumping:
                        self.jumping = True
                        self.jump_velocity = 15
                        self.create_particles(self.player.centerx, self.player.bottom, self.GREEN, 8)
                    elif self.double_jump_available and not self.double_jumping:
                        self.double_jumping = True
                        self.jump_velocity = 12
                        self.double_jump_available = False
                        self.create_particles(self.player.centerx, self.player.bottom, self.MAGENTA, 12)
                
                position, size, detected = eye_position, 0, eye_detected
                blink_detected = blink_detected
            else:
                palm_position, palm_size, palm_detected = self.detector.process_frame(frame)
                
                # Game logic based on palm position and size
                if palm_position == "right":
                    speed = 12 if self.speed_boost_active else 8
                    self.player.x = min(self.WIDTH - self.player.width, self.player.x + speed)
                elif palm_position == "left":
                    speed = 12 if self.speed_boost_active else 8
                    self.player.x = max(0, self.player.x - speed)
                elif palm_position == "center" and palm_size > 30000:  # Large palm = jump
                    if not self.jumping:
                        self.jumping = True
                        self.jump_velocity = 15
                        self.create_particles(self.player.centerx, self.player.bottom, self.GREEN, 8)
                    elif self.double_jump_available and not self.double_jumping:
                        self.double_jumping = True
                        self.jump_velocity = 12
                        self.double_jump_available = False
                        self.create_particles(self.player.centerx, self.player.bottom, self.MAGENTA, 12)
                
                position, size, detected = palm_position, palm_size, palm_detected
                blink_detected = False
            
            # Update game state
            self.handle_jump()
            self.spawn_obstacles()
            self.spawn_power_ups()
            
            if not self.update_obstacles():
                # Game over
                high_score = self.score_manager.save_high_score(self.score)
                print(f"Game Over! Final Score: {self.score}")
                
                game_over_action = self.game_over_screen.run(self.score, high_score)
                if game_over_action == "play_again":
                    self.reset_game()
                    continue
                elif game_over_action == "main_menu":
                    return "main_menu"
                elif game_over_action == "exit":
                    running = False
                break
            
            self.update_power_ups()
            self.update_power_up_timers(dt)
            self.update_particles()
            
            # Time-based scoring (every 10 seconds)
            current_time = time.time()
            time_alive = current_time - self.game_start_time
            if time_alive - self.last_time_score >= 10.0:
                self.score += 20
                self.last_time_score = time_alive
                print(f"Survival bonus! Score: {self.score}")
            
            self.draw(position, size, detected, blink_detected)
            
            # Show appropriate window title
            window_title = "Eye Detection" if self.control_mode == "eye" else "Palm Detection"
            cv2.imshow(window_title, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            target_fps = 30 if self.time_slow_active else 60
            self.clock.tick(target_fps)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        return "exit"

def main():
    print("Eye/Palm-Controlled Game")
    print("=" * 30)
    print("Choose your control method:")
    print("1. Eye Control - Look left/right to move, blink to jump")
    print("2. Palm Control - Move palm left/right to move, bring palm closer to jump")
    print()
    
    while True:
        choice = input("Enter 1 for Eye Control or 2 for Palm Control: ").strip()
        if choice == "1":
            control_mode = "eye"
            print("\nEye Control Selected!")
            print("Make sure your face is well-lit and visible to the camera.")
            print("Look left/right to move, blink to jump.")
            break
        elif choice == "2":
            control_mode = "palm"
            print("\nPalm Control Selected!")
            print("Move your palm left/right to control the character.")
            print("Bring your palm closer to the camera to jump.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print()
    game = PalmControlledGame(control_mode)
    game.run()

if __name__ == "__main__":
    main() 