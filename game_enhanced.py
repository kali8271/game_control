import cv2
import pygame
import sys
import numpy as np
import time
import random
import math
from collections import deque

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
        self.velocity_y += 0.5  # Gravity
        self.life -= 1
    
    def draw(self, screen):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = (*self.color, alpha)
            size = int(5 * (self.life / self.max_life))
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

class PowerUp:
    def __init__(self, x, y, power_type):
        self.rect = pygame.Rect(x, y, 30, 30)
        self.power_type = power_type
        self.collected = False
        self.animation_time = 0
        
        # Colors for different power-ups
        self.colors = {
            'double_jump': (255, 0, 255),  # Magenta
            'speed_boost': (255, 255, 0),  # Yellow
            'shield': (0, 255, 255),       # Cyan
            'time_slow': (128, 0, 128)     # Purple
        }
    
    def update(self):
        self.animation_time += 0.1
        self.rect.y += math.sin(self.animation_time) * 2
    
    def draw(self, screen):
        if not self.collected:
            color = self.colors.get(self.power_type, (255, 255, 255))
            pygame.draw.rect(screen, color, self.rect)
            # Draw power-up symbol
            if self.power_type == 'double_jump':
                pygame.draw.circle(screen, (255, 255, 255), self.rect.center, 8)
            elif self.power_type == 'speed_boost':
                pygame.draw.polygon(screen, (255, 255, 255), [
                    (self.rect.centerx - 8, self.rect.centery + 8),
                    (self.rect.centerx + 8, self.rect.centery),
                    (self.rect.centerx - 8, self.rect.centery - 8)
                ])

class AdvancedHandDetector:
    def __init__(self):
        # Enhanced skin detection ranges
        self.skin_ranges = [
            (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 50, 50], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            (np.array([0, 30, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
        ]
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)
        self.finger_history = deque(maxlen=5)
        
        # Performance optimization
        self.frame_skip = 2
        self.frame_count = 0
        
        # Gesture recognition
        self.gesture_types = ['none', 'fist', 'open_palm', 'peace', 'thumbs_up', 'point']
    
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
            return None, None
        
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
        
        return best_contour, mask
    
    def recognize_gesture(self, contour):
        """Advanced gesture recognition"""
        if contour is None or len(contour) < 5:
            return 'none'
        
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3:
            return 'none'
        
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 'none'
        
        # Count valid defects
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
        
        # Gesture classification
        if valid_defects == 0:
            return 'fist'
        elif valid_defects == 1:
            return 'point'
        elif valid_defects == 2:
            return 'peace'
        elif valid_defects == 3:
            return 'open_palm'
        elif valid_defects >= 4:
            return 'open_palm'
        
        return 'none'
    
    def count_fingers(self, contour):
        """Enhanced finger counting"""
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
        
        contour, mask = self.detect_hand_region(frame)
        
        finger_count = self.count_fingers(contour)
        gesture = self.recognize_gesture(contour)
        movement = self.detect_movement(contour, frame.shape[1])
        
        self.finger_history.append(finger_count)
        self.gesture_history.append(gesture)
        
        # Draw visualizations
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            
            # Draw gesture info
            cv2.putText(frame, f"Gesture: {gesture}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return self.get_smoothed_results()
    
    def get_smoothed_results(self):
        if not self.finger_history:
            return 0, "neutral", "none"
        
        finger_count = max(set(self.finger_history), key=self.finger_history.count)
        gesture = max(set(self.gesture_history), key=self.gesture_history.count) if self.gesture_history else "none"
        
        return finger_count, "neutral", gesture

class EnhancedGame:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Enhanced Gesture-Controlled Game")
        
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
        
        # Game state
        self.game_mode = "endless"  # endless, time_attack, gesture_challenge
        self.game_time = 0
        self.target_score = 1000
        
        # Performance
        self.clock = pygame.time.Clock()
        self.last_obstacle_time = time.time()
        self.last_power_up_time = time.time()
        self.obstacle_spawn_rate = 2.0
        self.power_up_spawn_rate = 5.0
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
    
    def create_particles(self, x, y, color, count=10):
        """Create particle effects"""
        for _ in range(count):
            velocity_x = random.uniform(-3, 3)
            velocity_y = random.uniform(-5, -1)
            particle = Particle(x, y, color, velocity_x, velocity_y)
            self.particles.append(particle)
    
    def handle_jump(self):
        """Enhanced jumping with double jump"""
        if self.jumping:
            self.player.y -= self.jump_velocity
            self.jump_velocity -= self.gravity
            
            if self.player.y >= 500:  # Ground level
                self.player.y = 500
                self.jumping = False
                self.double_jumping = False
                self.jump_velocity = 0
                self.double_jump_available = self.double_jump_available
    
    def spawn_obstacles(self):
        """Spawn obstacles with variety"""
        current_time = time.time()
        if current_time - self.last_obstacle_time > self.obstacle_spawn_rate:
            obstacle_type = random.choice(['normal', 'flying', 'ground'])
            
            if obstacle_type == 'normal':
                obstacle = pygame.Rect(self.WIDTH, 500, 30, 60)
            elif obstacle_type == 'flying':
                obstacle = pygame.Rect(self.WIDTH, 400, 30, 30)
            else:  # ground
                obstacle = pygame.Rect(self.WIDTH, 520, 30, 40)
            
            self.obstacles.append({'rect': obstacle, 'type': obstacle_type})
            self.last_obstacle_time = current_time
    
    def spawn_power_ups(self):
        """Spawn power-ups"""
        current_time = time.time()
        if current_time - self.last_power_up_time > self.power_up_spawn_rate:
            power_types = ['double_jump', 'speed_boost', 'shield', 'time_slow']
            power_type = random.choice(power_types)
            power_up = PowerUp(self.WIDTH, random.randint(300, 450), power_type)
            self.power_ups.append(power_up)
            self.last_power_up_time = current_time
    
    def update_obstacles(self):
        """Update obstacles and check collisions"""
        speed = self.game_speed * 2 if self.speed_boost_active else self.game_speed
        
        for obstacle_data in self.obstacles[:]:
            obstacle = obstacle_data['rect']
            obstacle.x -= speed
            
            # Check collision
            if self.player.colliderect(obstacle):
                if self.shield_active:
                    # Shield protects from one hit
                    self.shield_active = False
                    self.create_particles(obstacle.centerx, obstacle.centery, self.CYAN, 20)
                else:
                    return False  # Game over
            
            # Remove obstacles that are off screen
            if obstacle.x < -obstacle.width:
                self.obstacles.remove(obstacle_data)
                self.score += 10
        
        return True
    
    def update_power_ups(self):
        """Update power-ups and check collection"""
        for power_up in self.power_ups[:]:
            power_up.update()
            
            if self.player.colliderect(power_up.rect) and not power_up.collected:
                power_up.collected = True
                self.activate_power_up(power_up.power_type)
                self.create_particles(power_up.rect.centerx, power_up.rect.centery, 
                                   power_up.colors[power_up.power_type], 15)
                self.power_ups.remove(power_up)
            
            # Remove power-ups that are off screen
            elif power_up.rect.x < -power_up.rect.width:
                self.power_ups.remove(power_up)
    
    def activate_power_up(self, power_type):
        """Activate power-up effects"""
        if power_type == 'double_jump':
            self.double_jump_available = True
            self.power_up_timers['double_jump'] = 10.0  # 10 seconds
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
        """Update power-up timers"""
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
        """Update particle effects"""
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)
    
    def draw(self, finger_count, movement, gesture):
        """Enhanced drawing with effects"""
        self.screen.fill(self.WHITE)
        
        # Draw background
        pygame.draw.rect(self.screen, self.BLACK, (0, 560, self.WIDTH, 140))
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Draw player with effects
        player_color = self.GREEN
        if self.shield_active:
            player_color = self.CYAN
            # Draw shield effect
            pygame.draw.circle(self.screen, self.CYAN, self.player.center, 35, 3)
        
        pygame.draw.rect(self.screen, player_color, self.player)
        
        # Draw obstacles
        for obstacle_data in self.obstacles:
            obstacle = obstacle_data['rect']
            color = self.RED if obstacle_data['type'] == 'normal' else self.YELLOW
            pygame.draw.rect(self.screen, color, obstacle)
        
        # Draw power-ups
        for power_up in self.power_ups:
            power_up.draw(self.screen)
        
        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        finger_text = self.font.render(f"Fingers: {finger_count}", True, self.BLACK)
        self.screen.blit(finger_text, (10, 50))
        
        gesture_text = self.font.render(f"Gesture: {gesture}", True, self.BLACK)
        self.screen.blit(gesture_text, (10, 90))
        
        # Power-up status
        y_offset = 130
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
            "Open Palm = Special Move",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (self.WIDTH - 250, 10 + i * 25))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        detector = AdvancedHandDetector()
        cap = cv2.VideoCapture(0)
        
        print("Enhanced Gesture-Controlled Game Started!")
        print("Features: Power-ups, Advanced Gestures, Particle Effects!")
        
        running = True
        last_time = time.time()
        
        while running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Process camera frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            finger_count, movement, gesture = detector.process_frame(frame)
            
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
            
            # Special gesture actions
            if gesture == 'fist' and self.double_jump_available and not self.jumping:
                self.jumping = True
                self.jump_velocity = 15
                self.double_jump_available = False
                self.create_particles(self.player.centerx, self.player.bottom, self.MAGENTA, 12)
            
            # Update game state
            self.handle_jump()
            self.spawn_obstacles()
            self.spawn_power_ups()
            
            if not self.update_obstacles():
                print(f"Game Over! Final Score: {self.score}")
                break
            
            self.update_power_ups()
            self.update_power_up_timers(dt)
            self.update_particles()
            
            # Draw everything
            self.draw(finger_count, movement, gesture)
            
            # Show camera feed
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Enhanced Hand Detection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Adjust frame rate based on time slow
            target_fps = 30 if self.time_slow_active else 60
            self.clock.tick(target_fps)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = EnhancedGame()
    game.run() 