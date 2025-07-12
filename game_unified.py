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

class MediaPipeDetector:
    """MediaPipe-based hand detection (original method)"""
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tips = [4, 8, 12, 16, 20]
    
    def count_fingers(self, hand_landmarks):
        count = 0
        landmarks = hand_landmarks.landmark
        if landmarks[4].x < landmarks[3].x:
            count += 1
        for tip in self.finger_tips[1:]:
            if landmarks[tip].y < landmarks[tip - 2].y:
                count += 1
        return count
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        finger_count = 0
        movement = "neutral"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            finger_count = self.count_fingers(hand_landmarks)
            
            # Detect movement based on hand position
            landmarks = hand_landmarks.landmark
            wrist_x = landmarks[0].x
            frame_center = 0.5
            
            if wrist_x < frame_center - 0.1:
                movement = "left"
            elif wrist_x > frame_center + 0.1:
                movement = "right"
            elif finger_count >= 3:
                movement = "jump"
        
        cv2.putText(frame, f"MediaPipe - Fingers: {finger_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Movement: {movement}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return finger_count, movement, "mediapipe"

class OpenCVDetector:
    """OpenCV-based hand detection (enhanced method)"""
    def __init__(self):
        self.skin_ranges = [
            (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 50, 50], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 30, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
        ]
        
        self.gesture_history = deque(maxlen=5)
        self.finger_history = deque(maxlen=5)
        self.frame_skip = 2
        self.frame_count = 0
    
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
    
    def detect_hand_region(self, frame):
        mask = self.get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.5 <= aspect_ratio <= 2.0:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity < 0.85:
                    score = area * (1 - solidity)
                    if score > best_score:
                        best_score = score
                        best_contour = contour
        
        return best_contour
    
    def recognize_gesture(self, contour):
        if contour is None or len(contour) < 5:
            return 'none'
        
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3:
            return 'none'
        
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 'none'
        
        valid_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            
            if b * c == 0:
                continue
                
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            
            if angle <= np.pi / 2 and d > 8000:
                valid_defects += 1
        
        if valid_defects == 0:
            return 'fist'
        elif valid_defects == 1:
            return 'point'
        elif valid_defects == 2:
            return 'peace'
        elif valid_defects >= 3:
            return 'open_palm'
        
        return 'none'
    
    def count_fingers(self, contour):
        gesture = self.recognize_gesture(contour)
        
        if gesture == 'fist':
            return 0
        elif gesture == 'point':
            return 1
        elif gesture == 'peace':
            return 2
        elif gesture == 'open_palm':
            return 5
        
        return 0
    
    def detect_movement(self, contour, frame_width):
        if contour is None:
            return "neutral"
        
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        frame_center = frame_width // 2
        margin = 80
        
        if center_x < frame_center - margin:
            return "left"
        elif center_x > frame_center + margin:
            return "right"
        elif w * h > 25000:
            return "jump"
        else:
            return "neutral"
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return self.get_smoothed_results()
        
        frame = cv2.flip(frame, 1)
        contour = self.detect_hand_region(frame)
        
        finger_count = self.count_fingers(contour)
        gesture = self.recognize_gesture(contour)
        movement = self.detect_movement(contour, frame.shape[1])
        
        self.finger_history.append(finger_count)
        self.gesture_history.append(gesture)
        
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
        
        cv2.putText(frame, f"OpenCV - Fingers: {finger_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Movement: {movement}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.get_smoothed_results()
    
    def get_smoothed_results(self):
        if not self.finger_history:
            return 0, "neutral", "opencv"
        
        finger_count = max(set(self.finger_history), key=self.finger_history.count)
        gesture = max(set(self.gesture_history), key=self.gesture_history.count) if self.gesture_history else "none"
        
        return finger_count, "neutral", "opencv"

class Menu:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Gesture Game - Choose Detection Method")
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 32)
        self.title_font = pygame.font.Font(None, 64)
        
        self.selected_option = 0
        self.options = []
        
        # Build menu options based on available detection methods
        if MEDIAPIPE_AVAILABLE:
            self.options.append("MediaPipe Detection (Advanced)")
        self.options.append("OpenCV Detection (Enhanced)")
        self.options.append("Exit")
    
    def draw(self):
        self.screen.fill(self.WHITE)
        
        # Title
        title = self.title_font.render("Gesture-Controlled Game", True, self.BLACK)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.small_font.render("Choose your detection method:", True, self.GRAY)
        subtitle_rect = subtitle.get_rect(center=(self.WIDTH // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu options
        for i, option in enumerate(self.options):
            color = self.BLUE if i == self.selected_option else self.BLACK
            text = self.font.render(option, True, color)
            rect = text.get_rect(center=(self.WIDTH // 2, 250 + i * 60))
            self.screen.blit(text, rect)
            
            # Draw selection indicator
            if i == self.selected_option:
                pygame.draw.rect(self.screen, self.BLUE, 
                               (rect.left - 20, rect.centery - 5, 10, 10))
        
        # Instructions
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select",
            "Press ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.GRAY)
            self.screen.blit(text, (50, 450 + i * 30))
        
        # Detection method info
        if MEDIAPIPE_AVAILABLE:
            info_text = "MediaPipe: More accurate, requires MediaPipe library"
            info_color = self.GREEN
        else:
            info_text = "OpenCV: Works without additional libraries"
            info_color = self.RED
        
        info = self.small_font.render(info_text, True, info_color)
        self.screen.blit(info, (50, 550))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_option = (self.selected_option - 1) % len(self.options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % len(self.options)
                    elif event.key == pygame.K_RETURN:
                        return self.selected_option
                    elif event.key == pygame.K_ESCAPE:
                        return None
            
            self.draw()
            pygame.time.Clock().tick(60)
        
        return None

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
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        pygame.display.get_surface().blit(overlay, (0, 0))
        
        # Game Over title
        title = self.large_font.render("GAME OVER", True, self.RED)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 150))
        pygame.display.get_surface().blit(title, title_rect)
        
        # Final score
        score_text = self.font.render(f"Final Score: {final_score}", True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 250))
        pygame.display.get_surface().blit(score_text, score_rect)
        
        # High score (if available)
        if high_score and final_score >= high_score:
            high_score_text = self.font.render("NEW HIGH SCORE!", True, self.GREEN)
            high_score_rect = high_score_text.get_rect(center=(self.WIDTH // 2, 320))
            pygame.display.get_surface().blit(high_score_text, high_score_rect)
        elif high_score:
            high_score_text = self.small_font.render(f"High Score: {high_score}", True, self.GRAY)
            high_score_rect = high_score_text.get_rect(center=(self.WIDTH // 2, 320))
            pygame.display.get_surface().blit(high_score_text, high_score_rect)
        
        # Menu options
        for i, option in enumerate(self.options):
            color = self.BLUE if i == self.selected_option else self.WHITE
            text = self.font.render(option, True, color)
            rect = text.get_rect(center=(self.WIDTH // 2, 400 + i * 60))
            pygame.display.get_surface().blit(text, rect)
            
            # Draw selection indicator
            if i == self.selected_option:
                pygame.draw.rect(pygame.display.get_surface(), self.BLUE, 
                               (rect.left - 20, rect.centery - 5, 10, 10))
        
        # Instructions
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

class PauseMenu:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        self.font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 32)
        
        self.selected_option = 0
        self.options = ["Resume", "Restart", "Main Menu", "Exit"]
    
    def draw(self, current_score):
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        pygame.display.get_surface().blit(overlay, (0, 0))
        
        # Pause title
        title = self.font.render("PAUSED", True, self.WHITE)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 200))
        pygame.display.get_surface().blit(title, title_rect)
        
        # Current score
        score_text = self.small_font.render(f"Current Score: {current_score}", True, self.GRAY)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 280))
        pygame.display.get_surface().blit(score_text, score_rect)
        
        # Menu options
        for i, option in enumerate(self.options):
            color = self.BLUE if i == self.selected_option else self.WHITE
            text = self.font.render(option, True, color)
            rect = text.get_rect(center=(self.WIDTH // 2, 350 + i * 60))
            pygame.display.get_surface().blit(text, rect)
            
            # Draw selection indicator
            if i == self.selected_option:
                pygame.draw.rect(pygame.display.get_surface(), self.BLUE, 
                               (rect.left - 20, rect.centery - 5, 10, 10))
        
        # Instructions
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select, ESC to resume"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.GRAY)
            pygame.display.get_surface().blit(text, (50, self.HEIGHT - 80 + i * 30))
        
        pygame.display.flip()
    
    def run(self, current_score):
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
                        return "resume"
            
            self.draw(current_score)
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

class UnifiedGame:
    def __init__(self, detection_method):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f"Unified Gesture Game - {detection_method}")
        
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
        
        # Detection method
        self.detection_method = detection_method
        self.detector = None
        self.setup_detector()
        
        # UI Components
        self.game_over_screen = GameOverScreen(self.WIDTH, self.HEIGHT)
        self.pause_menu = PauseMenu(self.WIDTH, self.HEIGHT)
        self.score_manager = ScoreManager()
        
        # Game state
        self.paused = False
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
    
    def setup_detector(self):
        """Setup the appropriate detector based on selection"""
        if self.detection_method == 0 and MEDIAPIPE_AVAILABLE:
            self.detector = MediaPipeDetector()
            print("Using MediaPipe detection")
        else:
            self.detector = OpenCVDetector()
            print("Using OpenCV detection")
    
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
                self.score += 10
        
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
    
    def draw(self, finger_count, movement, gesture):
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
        
        method_text = self.font.render(f"Method: {self.detection_method.upper()}", True, self.BLACK)
        self.screen.blit(method_text, (10, 90 if high_score > 0 else 50))
        
        finger_text = self.font.render(f"Fingers: {finger_count}", True, self.BLACK)
        self.screen.blit(finger_text, (10, 130 if high_score > 0 else 90))
        
        if hasattr(self.detector, 'recognize_gesture'):
            gesture_text = self.font.render(f"Gesture: {gesture}", True, self.BLACK)
            self.screen.blit(gesture_text, (10, 170 if high_score > 0 else 130))
        
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
        
        # Instructions
        instructions = [
            "Controls:",
            "1 finger = Move Right",
            "2 fingers = Move Left", 
            "3+ fingers = Jump",
            "Fist = Double Jump (if available)",
            "Press 'p' to pause, 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (self.WIDTH - 250, 10 + i * 25))
        
        pygame.display.flip()
    
    def reset_game(self):
        """Reset game state for restart"""
        self.player = pygame.Rect(100, 500, 40, 60)
        self.obstacles = []
        self.power_ups = []
        self.particles = []
        self.score = 0
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
        self.paused = False
    
    def run(self):
        """Main game loop with enhanced UI"""
        cap = cv2.VideoCapture(0)
        
        print(f"Unified Game Started with {self.detection_method.upper()} detection!")
        print("Press 'p' to pause, 'q' to quit")
        
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
                    if event.key == pygame.K_p:
                        self.paused = True
                    elif event.key == pygame.K_q:
                        running = False
            
            # Handle pause menu
            if self.paused:
                pause_action = self.pause_menu.run(self.score)
                if pause_action == "resume":
                    self.paused = False
                elif pause_action == "restart":
                    self.reset_game()
                    self.paused = False
                elif pause_action == "main_menu":
                    return "main_menu"
                elif pause_action == "exit":
                    running = False
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            finger_count, movement, gesture = self.detector.process_frame(frame)
            
            # Game logic
            if movement == "right" or finger_count == 1:
                speed = 12 if self.speed_boost_active else 8
                self.player.x = min(self.WIDTH - self.player.width, self.player.x + speed)
            elif movement == "left" or finger_count == 2:
                speed = 12 if self.speed_boost_active else 8
                self.player.x = max(0, self.player.x - speed)
            elif movement == "jump" or finger_count >= 3:
                if not self.jumping:
                    self.jumping = True
                    self.jump_velocity = 15
                    self.create_particles(self.player.centerx, self.player.bottom, self.GREEN, 8)
                elif self.double_jump_available and not self.double_jumping:
                    self.double_jumping = True
                    self.jump_velocity = 12
                    self.double_jump_available = False
                    self.create_particles(self.player.centerx, self.player.bottom, self.MAGENTA, 12)
            
            # Special gesture actions (OpenCV only)
            if hasattr(self.detector, 'recognize_gesture') and gesture == 'fist' and self.double_jump_available and not self.jumping:
                self.jumping = True
                self.jump_velocity = 15
                self.double_jump_available = False
                self.create_particles(self.player.centerx, self.player.bottom, self.MAGENTA, 12)
            
            # Update game state
            self.handle_jump()
            self.spawn_obstacles()
            self.spawn_power_ups()
            
            if not self.update_obstacles():
                # Game over - show game over screen
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
            
            self.draw(finger_count, movement, gesture)
            
            cv2.imshow(f"Hand Detection - {self.detection_method.upper()}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            target_fps = 30 if self.time_slow_active else 60
            self.clock.tick(target_fps)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        return "exit"

def main():
    print("Gesture-Controlled Game Launcher")
    print("=" * 40)
    
    if MEDIAPIPE_AVAILABLE:
        print("✓ MediaPipe available - Advanced detection enabled")
    else:
        print("✗ MediaPipe not available - Using OpenCV detection")
    print()
    
    # Show menu
    menu = Menu()
    selection = menu.run()
    
    if selection is None:
        print("Exiting...")
        return
    
    if selection == len(menu.options) - 1:  # Exit option
        print("Exiting...")
        return
    
    # Determine detection method
    if selection == 0 and MEDIAPIPE_AVAILABLE:
        detection_method = "MediaPipe"
    else:
        detection_method = "OpenCV"
    
    # Start game loop
    while True:
        game = UnifiedGame(detection_method)
        result = game.run()
        
        if result == "main_menu":
            # Return to main menu
            selection = menu.run()
            if selection is None:
                print("Exiting...")
                break
            
            if selection == len(menu.options) - 1:  # Exit option
                print("Exiting...")
                break
            
            # Determine detection method
            if selection == 0 and MEDIAPIPE_AVAILABLE:
                detection_method = "MediaPipe"
            else:
                detection_method = "OpenCV"
        else:
            break

if __name__ == "__main__":
    main() 