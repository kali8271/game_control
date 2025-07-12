import cv2
import pygame
import sys
import numpy as np
import time
from collections import deque

class HandDetector:
    def __init__(self):
        # Optimized skin color ranges for different lighting conditions
        self.skin_ranges = [
            # Light skin
            (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            # Darker skin
            (np.array([0, 50, 50], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),
            # Alternative range
            (np.array([0, 30, 60], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
        ]
        
        # Smoothing for better detection
        self.finger_history = deque(maxlen=5)
        self.movement_history = deque(maxlen=3)
        
        # Performance optimization
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        
    def get_skin_mask(self, frame):
        """Create optimized skin mask using multiple ranges"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.skin_ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, range_mask)
        
        # Optimized morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_hand_region(self, frame):
        """Detect hand region with improved accuracy"""
        mask = self.get_skin_mask(frame)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the best hand contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:  # Too small
                continue
                
            # Calculate aspect ratio to favor hand-like shapes
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Hand typically has aspect ratio between 0.5 and 2.0
            if 0.5 <= aspect_ratio <= 2.0:
                # Calculate solidity (area / hull area) - hands are less solid than faces
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Hands have lower solidity than faces
                if solidity < 0.85:
                    score = area * (1 - solidity)  # Favor larger, less solid contours
                    if score > best_score:
                        best_score = score
                        best_contour = contour
        
        return best_contour, mask
    
    def count_fingers(self, contour):
        """Improved finger counting using convexity defects"""
        if contour is None or len(contour) < 5:
            return 0
        
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3:
            return 0
        
        # Get convexity defects
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        
        finger_count = 0
        valid_defects = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate angle using cosine rule
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            
            if b * c == 0:
                continue
                
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            
            # Filter defects based on angle and depth
            if angle <= np.pi / 2 and d > 8000:
                valid_defects.append((far, d))
                finger_count += 1
        
        # Sort defects by depth and take the top ones
        valid_defects.sort(key=lambda x: x[1], reverse=True)
        finger_count = min(len(valid_defects), 4)  # Max 4 defects = 5 fingers
        
        return finger_count + 1  # Add 1 for thumb
    
    def detect_movement(self, contour, frame_width):
        """Detect hand movement direction"""
        if contour is None:
            return "neutral"
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        
        # Calculate movement based on position
        frame_center = frame_width // 2
        margin = 80
        
        if center_x < frame_center - margin:
            return "left"
        elif center_x > frame_center + margin:
            return "right"
        elif w * h > 25000:  # Large hand area
            return "jump"
        else:
            return "neutral"
    
    def process_frame(self, frame):
        """Main processing function with performance optimization"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return self.get_smoothed_results()
        
        # Detect hand
        contour, mask = self.detect_hand_region(frame)
        
        # Count fingers
        finger_count = self.count_fingers(contour)
        self.finger_history.append(finger_count)
        
        # Detect movement
        movement = self.detect_movement(contour, frame.shape[1])
        self.movement_history.append(movement)
        
        # Draw visualizations
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
        
        return self.get_smoothed_results()
    
    def get_smoothed_results(self):
        """Get smoothed results from history"""
        if not self.finger_history:
            return 0, "neutral"
        
        # Get most common finger count
        finger_count = max(set(self.finger_history), key=self.finger_history.count)
        
        # Get most common movement
        movement = max(set(self.movement_history), key=self.movement_history.count) if self.movement_history else "neutral"
        
        return finger_count, movement

class Game:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Optimized Gesture-Controlled Game")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        
        # Game objects
        self.player = pygame.Rect(100, 400, 40, 60)
        self.obstacles = []
        self.score = 0
        self.game_speed = 5
        self.jumping = False
        self.jump_velocity = 0
        self.gravity = 0.8
        
        # Performance
        self.clock = pygame.time.Clock()
        self.last_obstacle_time = time.time()
        self.obstacle_spawn_rate = 2.0  # seconds
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def handle_jump(self):
        """Improved jumping physics"""
        if self.jumping:
            self.player.y -= self.jump_velocity
            self.jump_velocity -= self.gravity
            
            if self.player.y >= 400:  # Ground level
                self.player.y = 400
                self.jumping = False
                self.jump_velocity = 0
    
    def spawn_obstacle(self):
        """Spawn obstacles at regular intervals"""
        current_time = time.time()
        if current_time - self.last_obstacle_time > self.obstacle_spawn_rate:
            obstacle = pygame.Rect(self.WIDTH, 400, 30, 60)
            self.obstacles.append(obstacle)
            self.last_obstacle_time = current_time
    
    def update_obstacles(self):
        """Update obstacle positions and check collisions"""
        for obstacle in self.obstacles[:]:
            obstacle.x -= self.game_speed
            
            # Check collision
            if self.player.colliderect(obstacle):
                return False  # Game over
            
            # Remove obstacles that are off screen
            if obstacle.x < -obstacle.width:
                self.obstacles.remove(obstacle)
                self.score += 10
        
        return True  # Game continues
    
    def draw(self, finger_count, movement):
        """Draw game elements"""
        self.screen.fill(self.WHITE)
        
        # Draw ground
        pygame.draw.rect(self.screen, self.BLACK, (0, 460, self.WIDTH, 140))
        
        # Draw player
        pygame.draw.rect(self.screen, self.GREEN, self.player)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.RED, obstacle)
        
        # Draw UI
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        finger_text = self.font.render(f"Fingers: {finger_count}", True, self.BLACK)
        self.screen.blit(finger_text, (10, 50))
        
        movement_text = self.font.render(f"Movement: {movement}", True, self.BLACK)
        self.screen.blit(movement_text, (10, 90))
        
        # Instructions
        instructions = [
            "Controls:",
            "1 finger = Move Right",
            "2 fingers = Move Left", 
            "3+ fingers = Jump",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.BLACK)
            self.screen.blit(text, (self.WIDTH - 200, 10 + i * 25))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        detector = HandDetector()
        cap = cv2.VideoCapture(0)
        
        print("Optimized Gesture-Controlled Game Started!")
        print("Controls: 1 finger=Right, 2 fingers=Left, 3+ fingers=Jump")
        
        running = True
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Process camera frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            finger_count, movement = detector.process_frame(frame)
            
            # Game logic
            if movement == "right" or finger_count == 1:
                self.player.x = min(self.WIDTH - self.player.width, self.player.x + 8)
            elif movement == "left" or finger_count == 2:
                self.player.x = max(0, self.player.x - 8)
            elif movement == "jump" or finger_count >= 3:
                if not self.jumping:
                    self.jumping = True
                    self.jump_velocity = 15
            
            # Update game state
            self.handle_jump()
            self.spawn_obstacle()
            
            if not self.update_obstacles():
                print(f"Game Over! Final Score: {self.score}")
                break
            
            # Draw everything
            self.draw(finger_count, movement)
            
            # Show camera feed
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Movement: {movement}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Detection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.clock.tick(60)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run() 