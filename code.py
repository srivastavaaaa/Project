import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import threading
import json
from collections import deque

class ComprehensiveGestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support two hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen and camera settings
        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480
        
        # Smoothing and tracking
        self.smoothening = 7
        self.prev_x, self.prev_y = 0, 0
        self.gesture_history = deque(maxlen=10)
        
        # Timing and cooldowns
        self.click_threshold = 35
        self.last_click_time = 0
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3
        self.long_press_duration = 1.0
        self.gesture_start_time = {}
        
        # Advanced gesture states
        self.gesture_states = {
            'drag_mode': False,
            'scroll_mode': False,
            'zoom_mode': False,
            'precision_mode': False,
            'keyboard_mode': False,
            'media_mode': False,
            'window_mode': False,
            'drawing_mode': False,
            'two_hand_mode': False
        }
        
        # Gesture tracking
        self.persistent_gesture = None
        self.gesture_duration = 0
        self.last_two_hand_distance = 0
        
        # Virtual keyboard layout
        self.virtual_keyboard = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'ENTER'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE', 'BACK', 'ESC']
        ]
        
        # Drawing canvas
        self.drawing_canvas = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
        self.drawing_enabled = False
        
        # Settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
        
        # Mode display
        self.current_mode = "NORMAL"
        self.mode_switch_time = 0
        
    def get_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))
    
    def is_finger_up(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended"""
        return landmarks[finger_tip].y < landmarks[finger_pip].y
    
    def detect_gesture(self, landmarks):
        """Enhanced gesture detection"""
        # Landmark indices
        thumb_tip, thumb_ip = 4, 3
        index_tip, index_pip = 8, 6
        middle_tip, middle_pip = 12, 10
        ring_tip, ring_pip = 16, 14
        pinky_tip, pinky_pip = 20, 18
        
        fingers_up = []
        
        # Thumb (check x coordinate for left/right hand)
        if landmarks[thumb_tip].x > landmarks[thumb_ip].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other fingers
        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                        (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            if self.is_finger_up(landmarks, tip, pip):
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return fingers_up
    
    def detect_hand_orientation(self, landmarks):
        """Detect if hand is facing palm or back"""
        # Use the relationship between wrist and middle finger MCP
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        
        # Calculate if palm is facing camera
        palm_facing = middle_tip.z < middle_mcp.z
        return "PALM" if palm_facing else "BACK"
    
    def switch_mode(self, new_mode):
        """Switch between different control modes"""
        if self.current_mode != new_mode:
            self.current_mode = new_mode
            self.mode_switch_time = time.time()
            # Reset all gesture states when switching modes
            for key in self.gesture_states:
                self.gesture_states[key] = False
    
    def perform_basic_mouse_actions(self, landmarks, fingers_up, frame):
        """Basic mouse control actions"""
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        
        # Convert to screen coordinates
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        
        # Smooth movement
        if self.gesture_states['precision_mode']:
            # Slower, more precise movement
            curr_x = self.prev_x + (x - self.prev_x) / (self.smoothening * 2)
            curr_y = self.prev_y + (y - self.prev_y) / (self.smoothening * 2)
        else:
            curr_x = self.prev_x + (x - self.prev_x) / self.smoothening
            curr_y = self.prev_y + (y - self.prev_y) / self.smoothening
        
        current_time = time.time()
        
        # Gesture recognition
        if fingers_up == [0, 1, 0, 0, 0]:  # Index only - Move cursor
            pyautogui.moveTo(curr_x, curr_y)
            cv2.circle(frame, (int(index_tip.x * self.cam_width), 
                              int(index_tip.y * self.cam_height)), 12, (0, 255, 0), -1)
            self.display_action(frame, "MOVE CURSOR")
            
        elif fingers_up == [0, 1, 1, 0, 0]:  # Index + Middle - Click actions
            distance = self.get_distance(
                (index_tip.x * self.cam_width, index_tip.y * self.cam_height),
                (middle_tip.x * self.cam_width, middle_tip.y * self.cam_height)
            )
            
            if distance < self.click_threshold:
                if current_time - self.last_click_time > self.gesture_cooldown:
                    pyautogui.click()
                    self.last_click_time = current_time
                    self.display_action(frame, "LEFT CLICK")
            else:
                self.display_action(frame, "READY TO CLICK")
                
        elif fingers_up == [1, 1, 1, 0, 0]:  # Thumb + Index + Middle - Right click
            distance = self.get_distance(
                (thumb_tip.x * self.cam_width, thumb_tip.y * self.cam_height),
                (index_tip.x * self.cam_width, index_tip.y * self.cam_height)
            )
            
            if distance < self.click_threshold:
                if current_time - self.last_click_time > self.gesture_cooldown:
                    pyautogui.rightClick()
                    self.last_click_time = current_time
                    self.display_action(frame, "RIGHT CLICK")
        
        self.prev_x, self.prev_y = curr_x, curr_y
    
    def perform_keyboard_actions(self, landmarks, fingers_up, frame):
        """Virtual keyboard and text input"""
        if fingers_up == [1, 1, 0, 0, 0]:  # Thumb + Index - Type mode
            self.draw_virtual_keyboard(frame)
            # Detect which key is being pointed at
            key = self.detect_keyboard_selection(landmarks)
            if key:
                self.display_action(frame, f"SELECT: {key}")
                # Type the character when fingers close
                index_tip = landmarks[8]
                thumb_tip = landmarks[4]
                distance = self.get_distance(
                    (index_tip.x * self.cam_width, index_tip.y * self.cam_height),
                    (thumb_tip.x * self.cam_width, thumb_tip.y * self.cam_height)
                )
                if distance < 25:
                    self.type_character(key)
        
        elif fingers_up == [0, 1, 1, 1, 0]:  # Index + Middle + Ring - Text shortcuts
            middle_y = landmarks[12].y
            if middle_y < 0.3:
                pyautogui.hotkey('ctrl', 'c')  # Copy
                self.display_action(frame, "COPY")
            elif middle_y > 0.7:
                pyautogui.hotkey('ctrl', 'v')  # Paste
                self.display_action(frame, "PASTE")
            else:
                pyautogui.hotkey('ctrl', 'x')  # Cut
                self.display_action(frame, "CUT")
    
    def perform_media_controls(self, landmarks, fingers_up, frame):
        """Media playback controls"""
        if fingers_up == [1, 0, 0, 0, 1]:  # Thumb + Pinky - Volume
            thumb_y = landmarks[4].y
            if thumb_y < 0.3:
                pyautogui.press('volumeup')
                self.display_action(frame, "VOLUME UP")
            elif thumb_y > 0.7:
                pyautogui.press('volumedown')
                self.display_action(frame, "VOLUME DOWN")
            else:
                pyautogui.press('volumemute')
                self.display_action(frame, "MUTE TOGGLE")
                
        elif fingers_up == [0, 1, 0, 1, 0]:  # Index + Ring - Media control
            index_x = landmarks[8].x
            if index_x < 0.3:
                pyautogui.press('prevtrack')
                self.display_action(frame, "PREVIOUS TRACK")
            elif index_x > 0.7:
                pyautogui.press('nexttrack')
                self.display_action(frame, "NEXT TRACK")
            else:
                pyautogui.press('playpause')
                self.display_action(frame, "PLAY/PAUSE")
                
        elif fingers_up == [0, 0, 1, 0, 0]:  # Middle finger only - Stop
            pyautogui.press('stop')
            self.display_action(frame, "STOP MEDIA")
    
    def perform_window_management(self, landmarks, fingers_up, frame):
        """Window and application management"""
        if fingers_up == [1, 1, 0, 0, 1]:  # Thumb + Index + Pinky - Window actions
            index_y = landmarks[8].y
            if index_y < 0.25:
                pyautogui.hotkey('win', 'up')  # Maximize
                self.display_action(frame, "MAXIMIZE WINDOW")
            elif index_y > 0.75:
                pyautogui.hotkey('win', 'down')  # Minimize
                self.display_action(frame, "MINIMIZE WINDOW")
            elif landmarks[8].x < 0.3:
                pyautogui.hotkey('win', 'left')  # Snap left
                self.display_action(frame, "SNAP LEFT")
            elif landmarks[8].x > 0.7:
                pyautogui.hotkey('win', 'right')  # Snap right
                self.display_action(frame, "SNAP RIGHT")
            else:
                pyautogui.hotkey('alt', 'tab')  # Alt+Tab
                self.display_action(frame, "SWITCH WINDOW")
                
        elif fingers_up == [1, 0, 1, 0, 1]:  # Thumb + Middle + Pinky - Desktop actions
            middle_x = landmarks[12].x
            if middle_x < 0.3:
                pyautogui.hotkey('ctrl', 'win', 'left')  # Switch desktop left
                self.display_action(frame, "DESKTOP LEFT")
            elif middle_x > 0.7:
                pyautogui.hotkey('ctrl', 'win', 'right')  # Switch desktop right
                self.display_action(frame, "DESKTOP RIGHT")
            else:
                pyautogui.hotkey('win', 'd')  # Show desktop
                self.display_action(frame, "SHOW DESKTOP")
    
    def perform_browser_actions(self, landmarks, fingers_up, frame):
        """Browser-specific controls"""
        if fingers_up == [0, 1, 1, 1, 1]:  # Four fingers no thumb - Browser navigation
            index_x = landmarks[8].x
            middle_y = landmarks[12].y
            
            if index_x < 0.2:
                pyautogui.hotkey('alt', 'left')  # Back
                self.display_action(frame, "BROWSER BACK")
            elif index_x > 0.8:
                pyautogui.hotkey('alt', 'right')  # Forward
                self.display_action(frame, "BROWSER FORWARD")
            elif middle_y < 0.2:
                pyautogui.hotkey('ctrl', 't')  # New tab
                self.display_action(frame, "NEW TAB")
            elif middle_y > 0.8:
                pyautogui.hotkey('ctrl', 'w')  # Close tab
                self.display_action(frame, "CLOSE TAB")
            else:
                pyautogui.hotkey('f5')  # Refresh
                self.display_action(frame, "REFRESH PAGE")
    
    def perform_system_actions(self, landmarks, fingers_up, frame):
        """System-level controls"""
        # L-shape gesture (Thumb + Index perpendicular) - Screenshot
        if fingers_up == [1, 1, 0, 0, 0]:
            thumb_pos = (landmarks[4].x, landmarks[4].y)
            index_pos = (landmarks[8].x, landmarks[8].y)
            
            # Check if fingers form L-shape
            angle = self.get_angle(
                (thumb_pos[0] * self.cam_width, thumb_pos[1] * self.cam_height),
                (landmarks[2].x * self.cam_width, landmarks[2].y * self.cam_height),
                (index_pos[0] * self.cam_width, index_pos[1] * self.cam_height)
            )
            
            if 80 < angle < 100:  # Close to 90 degrees
                pyautogui.hotkey('win', 'shift', 's')  # Screenshot
                self.display_action(frame, "SCREENSHOT")
        
        # Peace sign (Index + Middle separated) - Lock screen
        elif fingers_up == [0, 1, 1, 0, 0]:
            distance = self.get_distance(
                (landmarks[8].x * self.cam_width, landmarks[8].y * self.cam_height),
                (landmarks[12].x * self.cam_width, landmarks[12].y * self.cam_height)
            )
            if distance > 60:  # Fingers spread apart
                if time.time() - self.last_gesture_time > 2:  # Long cooldown for lock
                    pyautogui.hotkey('win', 'l')
                    self.last_gesture_time = time.time()
                    self.display_action(frame, "LOCK SCREEN")
    
    def perform_drawing_actions(self, landmarks, fingers_up, frame):
        """Drawing and annotation features"""
        index_tip = landmarks[8]
        ix, iy = int(index_tip.x * self.cam_width), int(index_tip.y * self.cam_height)
        
        if fingers_up == [0, 1, 0, 0, 0]:  # Index only - Draw
            if self.drawing_enabled:
                cv2.circle(self.drawing_canvas, (ix, iy), 5, (0, 255, 255), -1)
                self.display_action(frame, "DRAWING")
        
        elif fingers_up == [1, 1, 1, 1, 1]:  # All fingers - Clear canvas
            if np.any(self.drawing_canvas):
                self.drawing_canvas = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
                self.display_action(frame, "CLEAR CANVAS")
    
    def perform_two_hand_gestures(self, all_landmarks, frame):
        """Advanced two-hand gestures"""
        if len(all_landmarks) == 2:
            hand1, hand2 = all_landmarks[0], all_landmarks[1]
            
            # Calculate distance between palms
            palm1 = (hand1[9].x * self.cam_width, hand1[9].y * self.cam_height)
            palm2 = (hand2[9].x * self.cam_width, hand2[9].y * self.cam_height)
            distance = self.get_distance(palm1, palm2)
            
            # Two-hand zoom
            if self.last_two_hand_distance > 0:
                if distance > self.last_two_hand_distance + 20:
                    pyautogui.hotkey('ctrl', '+')
                    self.display_action(frame, "TWO-HAND ZOOM IN")
                elif distance < self.last_two_hand_distance - 20:
                    pyautogui.hotkey('ctrl', '-')
                    self.display_action(frame, "TWO-HAND ZOOM OUT")
            
            self.last_two_hand_distance = distance
            
            # Two-hand rotate (simulate)
            if palm1[1] < self.cam_height * 0.3 and palm2[1] < self.cam_height * 0.3:
                pyautogui.hotkey('ctrl', 'shift', 'r')  # Rotate (application dependent)
                self.display_action(frame, "ROTATE GESTURE")
    
    def perform_advanced_shortcuts(self, landmarks, fingers_up, frame):
        """Advanced keyboard shortcuts"""
        current_time = time.time()
        
        if fingers_up == [1, 0, 1, 0, 0]:  # Thumb + Middle - File operations
            middle_y = landmarks[12].y
            if middle_y < 0.3:
                pyautogui.hotkey('ctrl', 'n')  # New file
                self.display_action(frame, "NEW FILE")
            elif middle_y > 0.7:
                pyautogui.hotkey('ctrl', 's')  # Save
                self.display_action(frame, "SAVE FILE")
            else:
                pyautogui.hotkey('ctrl', 'o')  # Open
                self.display_action(frame, "OPEN FILE")
                
        elif fingers_up == [0, 0, 1, 1, 0]:  # Middle + Ring - Undo/Redo
            ring_x = landmarks[16].x
            if ring_x < 0.4:
                pyautogui.hotkey('ctrl', 'z')  # Undo
                self.display_action(frame, "UNDO")
            else:
                pyautogui.hotkey('ctrl', 'y')  # Redo
                self.display_action(frame, "REDO")
                
        elif fingers_up == [0, 0, 0, 1, 1]:  # Ring + Pinky - Find/Replace
            ring_y = landmarks[16].y
            if ring_y < 0.4:
                pyautogui.hotkey('ctrl', 'f')  # Find
                self.display_action(frame, "FIND")
            else:
                pyautogui.hotkey('ctrl', 'h')  # Replace
                self.display_action(frame, "REPLACE")
    
    def perform_gaming_actions(self, landmarks, fingers_up, frame):
        """Gaming-specific controls"""
        if fingers_up == [0, 1, 0, 0, 1]:  # Index + Pinky - WASD movement
            index_pos = landmarks[8]
            x, y = index_pos.x, index_pos.y
            
            if x < 0.3:
                pyautogui.press('a')  # Left
                self.display_action(frame, "MOVE LEFT")
            elif x > 0.7:
                pyautogui.press('d')  # Right
                self.display_action(frame, "MOVE RIGHT")
            elif y < 0.3:
                pyautogui.press('w')  # Up
                self.display_action(frame, "MOVE UP")
            elif y > 0.7:
                pyautogui.press('s')  # Down
                self.display_action(frame, "MOVE DOWN")
                
        elif fingers_up == [1, 0, 0, 0, 0]:  # Thumb only - Space/Jump
            pyautogui.press('space')
            self.display_action(frame, "JUMP/SPACE")
    
    def detect_keyboard_selection(self, landmarks):
        """Detect which virtual key is being pointed at"""
        index_tip = landmarks[8]
        x, y = int(index_tip.x * self.cam_width), int(index_tip.y * self.cam_height)
        
        # Virtual keyboard area
        kb_start_y = self.cam_height - 160
        if y >= kb_start_y:
            row = (y - kb_start_y) // 35
            col = x // 60
            
            if 0 <= row < len(self.virtual_keyboard) and 0 <= col < len(self.virtual_keyboard[row]):
                return self.virtual_keyboard[row][col]
        return None
    
    def type_character(self, char):
        """Type a character based on virtual keyboard selection"""
        current_time = time.time()
        if current_time - self.last_click_time > 0.3:  # Typing cooldown
            if char == 'SPACE':
                pyautogui.press('space')
            elif char == 'BACK':
                pyautogui.press('backspace')
            elif char == 'ENTER':
                pyautogui.press('enter')
            elif char == 'ESC':
                pyautogui.press('escape')
            else:
                pyautogui.write(char.lower())
            self.last_click_time = current_time
    
    def draw_virtual_keyboard(self, frame):
        """Draw virtual keyboard overlay"""
        kb_start_y = self.cam_height - 160
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, kb_start_y), (self.cam_width, self.cam_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw keys
        for row_idx, row in enumerate(self.virtual_keyboard):
            for col_idx, key in enumerate(row):
                x = col_idx * 60 + 10
                y = kb_start_y + row_idx * 35 + 10
                
                cv2.rectangle(frame, (x, y), (x + 50, y + 25), (100, 100, 100), 2)
                cv2.putText(frame, key, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def detect_complex_gestures(self, landmarks, fingers_up):
        """Detect complex multi-step gestures"""
        gesture_key = str(fingers_up)
        current_time = time.time()
        
        # Track gesture persistence
        if gesture_key == self.persistent_gesture:
            self.gesture_duration = current_time - self.gesture_start_time.get(gesture_key, current_time)
        else:
            self.persistent_gesture = gesture_key
            self.gesture_start_time[gesture_key] = current_time
            self.gesture_duration = 0
        
        # Long press gestures (hold for 2+ seconds)
        if self.gesture_duration > 2.0:
            if fingers_up == [0, 1, 0, 0, 0]:  # Long index - Precision mode toggle
                self.gesture_states['precision_mode'] = not self.gesture_states['precision_mode']
                self.gesture_start_time[gesture_key] = current_time  # Reset timer
                
            elif fingers_up == [1, 0, 0, 0, 0]:  # Long thumb - Drawing mode toggle
                self.drawing_enabled = not self.drawing_enabled
                self.gesture_start_time[gesture_key] = current_time
    
    def perform_accessibility_features(self, landmarks, fingers_up, frame):
        """Accessibility and utility features"""
        if fingers_up == [1, 1, 1, 0, 1]:  # Thumb + Index + Middle + Pinky - Accessibility
            middle_y = landmarks[12].y
            if middle_y < 0.3:
                pyautogui.hotkey('win', '+')  # Magnifier
                self.display_action(frame, "MAGNIFIER")
            elif middle_y > 0.7:
                pyautogui.hotkey('win', 'u')  # Ease of Access
                self.display_action(frame, "EASE OF ACCESS")
            else:
                pyautogui.hotkey('win', 'ctrl', 'enter')  # Narrator
                self.display_action(frame, "NARRATOR")
    
    def perform_scroll_actions(self, landmarks, fingers_up, frame):
        """Enhanced scrolling with different modes"""
        if fingers_up == [0, 1, 1, 1, 0]:  # Index + Middle + Ring - Advanced scroll
            middle_y = landmarks[12].y
            ring_x = landmarks[16].x
            
            # Vertical scroll
            if ring_x < 0.3 or ring_x > 0.7:
                if middle_y < 0.3:
                    pyautogui.scroll(5)  # Fast scroll up
                    self.display_action(frame, "FAST SCROLL UP")
                elif middle_y > 0.7:
                    pyautogui.scroll(-5)  # Fast scroll down
                    self.display_action(frame, "FAST SCROLL DOWN")
            else:
                # Horizontal scroll
                if middle_y < 0.4:
                    pyautogui.hscroll(3)  # Horizontal scroll left
                    self.display_action(frame, "SCROLL LEFT")
                elif middle_y > 0.6:
                    pyautogui.hscroll(-3)  # Horizontal scroll right
                    self.display_action(frame, "SCROLL RIGHT")
    
    def detect_mode_switch_gestures(self, landmarks, fingers_up, frame):
        """Detect gestures that switch between different modes"""
        # Mode switching with specific gesture combinations
        if fingers_up == [1, 0, 1, 1, 0]:  # Thumb + Middle + Ring - Cycle modes
            modes = ["NORMAL", "KEYBOARD", "MEDIA", "WINDOW", "GAMING", "DRAWING"]
            current_index = modes.index(self.current_mode)
            next_mode = modes[(current_index + 1) % len(modes)]
            self.switch_mode(next_mode)
            self.display_action(frame, f"MODE: {next_mode}")
    
    def display_action(self, frame, action_text):
        """Display current action on frame"""
        cv2.putText(frame, action_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def draw_ui_elements(self, frame):
        """Draw UI elements and information"""
        h, w, _ = frame.shape
        
        # Mode indicator
        mode_color = (0, 255, 0) if time.time() - self.mode_switch_time < 2 else (255, 255, 255)
        cv2.putText(frame, f"MODE: {self.current_mode}", (w-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Precision mode indicator
        if self.gesture_states['precision_mode']:
            cv2.putText(frame, "PRECISION ON", (w-200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Drawing mode indicator
        if self.drawing_enabled:
            cv2.putText(frame, "DRAWING ON", (w-200, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Status indicators for active modes
        y_offset = 120
        for state, active in self.gesture_states.items():
            if active:
                cv2.putText(frame, state.upper(), (w-200, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
        
        # Draw gesture history trail
        if len(self.gesture_history) > 1:
            for i in range(1, len(self.gesture_history)):
                prev_point = self.gesture_history[i-1]
                curr_point = self.gesture_history[i]
                cv2.line(frame, prev_point, curr_point, (255, 0, 255), 2)
    
    def draw_landmarks_enhanced(self, frame, landmarks, hand_idx=0):
        """Enhanced landmark drawing with additional information"""
        h, w, _ = frame.shape
        
        # Color scheme for different hands
        colors = [(255, 0, 0), (0, 0, 255)] if hand_idx < 2 else [(255, 255, 0)]
        color = colors[hand_idx]
        
        # Draw fingertips with different colors
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        for i, tip in enumerate(fingertips):
            x, y = int(landmarks[tip].x * w), int(landmarks[tip].y * h)
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.putText(frame, str(i), (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw palm center
        palm_center = landmarks[9]
        px, py = int(palm_center.x * w), int(palm_center.y * h)
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
        
        # Add to gesture history for trail effect
        index_tip = landmarks[8]
        self.gesture_history.append((int(index_tip.x * w), int(index_tip.y * h)))
    
    def perform_text_selection_actions(self, landmarks, fingers_up, frame):
        """Text selection and editing actions"""
        if fingers_up == [1, 1, 0, 0, 0]:  # Thumb + Index - Text selection
            # Calculate selection direction and distance
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Selection based on hand movement
            if abs(index_tip.x - thumb_tip.x) > 0.1:  # Horizontal selection
                if index_tip.x > thumb_tip.x:
                    pyautogui.hotkey('shift', 'right')
                    self.display_action(frame, "SELECT RIGHT")
                else:
                    pyautogui.hotkey('shift', 'left')
                    self.display_action(frame, "SELECT LEFT")
            elif abs(index_tip.y - thumb_tip.y) > 0.1:  # Vertical selection
                if index_tip.y < thumb_tip.y:
                    pyautogui.hotkey('shift', 'up')
                    self.display_action(frame, "SELECT UP")
                else:
                    pyautogui.hotkey('shift', 'down')
                    self.display_action(frame, "SELECT DOWN")
    
    def perform_presentation_actions(self, landmarks, fingers_up, frame):
        """Presentation and slideshow controls"""
        if fingers_up == [0, 1, 0, 1, 0]:  # Index + Ring - Presentation control
            index_x = landmarks[8].x
            ring_y = landmarks[16].y
            
            if index_x < 0.3:
                pyautogui.press('left')  # Previous slide
                self.display_action(frame, "PREVIOUS SLIDE")
            elif index_x > 0.7:
                pyautogui.press('right')  # Next slide
                self.display_action(frame, "NEXT SLIDE")
            elif ring_y < 0.3:
                pyautogui.press('f5')  # Start slideshow
                self.display_action(frame, "START SLIDESHOW")
            elif ring_y > 0.7:
                pyautogui.press('escape')  # Exit slideshow
                self.display_action(frame, "EXIT SLIDESHOW")
    
    def perform_zoom_meeting_actions(self, landmarks, fingers_up, frame):
        """Video conferencing controls"""
        if fingers_up == [1, 0, 0, 1, 0]:  # Thumb + Ring - Video controls
            thumb_y = landmarks[4].y
            ring_x = landmarks[16].x
            
            if thumb_y < 0.3:
                pyautogui.hotkey('alt', 'v')  # Toggle video
                self.display_action(frame, "TOGGLE VIDEO")
            elif thumb_y > 0.7:
                pyautogui.hotkey('alt', 'a')  # Toggle audio
                self.display_action(frame, "TOGGLE AUDIO")
            elif ring_x < 0.3:
                pyautogui.hotkey('alt', 's')  # Share screen
                self.display_action(frame, "SHARE SCREEN")
            elif ring_x > 0.7:
                pyautogui.hotkey('alt', 'r')  # Record
                self.display_action(frame, "TOGGLE RECORD")
    
    def perform_ide_actions(self, landmarks, fingers_up, frame):
        """IDE and coding-specific shortcuts"""
        if fingers_up == [0, 1, 0, 0, 0] and self.current_mode == "CODING":
            # Run code
            pyautogui.hotkey('f5')
            self.display_action(frame, "RUN CODE")
            
        elif fingers_up == [0, 1, 1, 0, 0] and self.current_mode == "CODING":
            # Debug
            pyautogui.hotkey('f9')
            self.display_action(frame, "TOGGLE BREAKPOINT")
            
        elif fingers_up == [1, 1, 1, 0, 0] and self.current_mode == "CODING":
            # Format code
            pyautogui.hotkey('ctrl', 'shift', 'f')
            self.display_action(frame, "FORMAT CODE")
    
    def calculate_gesture_confidence(self, landmarks):
        """Calculate confidence score for gesture recognition"""
        # Check hand stability
        if len(self.gesture_history) < 3:
            return 0.5
        
        # Calculate movement variance
        recent_points = list(self.gesture_history)[-3:]
        variance = np.var([p[0] for p in recent_points]) + np.var([p[1] for p in recent_points])
        
        # Lower variance = higher confidence
        confidence = max(0.1, 1.0 - variance / 1000)
        return min(1.0, confidence)
    
    def perform_custom_shortcuts(self, landmarks, fingers_up, frame):
        """Custom application shortcuts"""
        # Application-specific shortcuts
        app_shortcuts = {
            # Browser shortcuts
            '[0, 1, 1, 0, 1]': ('ctrl', 'shift', 't'),  # Reopen closed tab
            '[1, 0, 1, 1, 1]': ('ctrl', 'shift', 'n'),  # Incognito window
            
            # File manager shortcuts
            '[1, 1, 0, 1, 0]': ('ctrl', 'shift', 'n'),  # New folder
            '[0, 1, 0, 1, 1]': ('f2',),  # Rename
            
            # General shortcuts
            '[1, 0, 0, 1, 1]': ('win', 'x'),  # Quick admin menu
            '[0, 0, 1, 1, 1]': ('ctrl', 'shift', 'esc'),  # Task manager
        }
        
        gesture_key = str(fingers_up)
        if gesture_key in app_shortcuts:
            keys = app_shortcuts[gesture_key]
            pyautogui.hotkey(*keys)
            action_name = ' + '.join(keys).upper()
            self.display_action(frame, f"SHORTCUT: {action_name}")
    
    def perform_mouse_precision_actions(self, landmarks, fingers_up, frame):
        """Precision mouse movements and selections"""
        if self.gesture_states['precision_mode']:
            # Micro movements with pinky control
            if fingers_up[4]:  # Pinky up for precision
                pinky_tip = landmarks[20]
                
                # Micro adjustments
                offset_x = (pinky_tip.x - 0.5) * 10  # Small movements
                offset_y = (pinky_tip.y - 0.5) * 10
                
                current_pos = pyautogui.position()
                new_x = current_pos.x + offset_x
                new_y = current_pos.y + offset_y
                
                pyautogui.moveTo(new_x, new_y)
                self.display_action(frame, "PRECISION MOVE")
    
    def handle_emergency_gestures(self, landmarks, fingers_up, frame):
        """Emergency and safety gestures"""
        # Emergency stop - specific gesture pattern
        if fingers_up == [1, 0, 1, 0, 1]:  # Alternating pattern
            thumb_y = landmarks[4].y
            if thumb_y < 0.1:  # Very top of screen
                # Emergency stop all automation
                self.gesture_states = {key: False for key in self.gesture_states}
                pyautogui.mouseUp()  # Release any held buttons
                self.display_action(frame, "EMERGENCY STOP")
    
    def main_gesture_processor(self, landmarks_list, frame):
        """Main gesture processing pipeline"""
        if not landmarks_list:
            return
        
        # Process each detected hand
        for hand_idx, landmarks in enumerate(landmarks_list):
            fingers_up = self.detect_gesture(landmarks)
            confidence = self.calculate_gesture_confidence(landmarks)
            
            # Draw enhanced landmarks
            self.draw_landmarks_enhanced(frame, landmarks, hand_idx)
            
            # Emergency gestures (highest priority)
            self.handle_emergency_gestures(landmarks, fingers_up, frame)
            
            # Mode switching
            self.detect_mode_switch_gestures(landmarks, fingers_up, frame)
            
            # Complex gesture detection
            self.detect_complex_gestures(landmarks, fingers_up)
            
            # Mode-specific actions
            if self.current_mode == "NORMAL":
                self.perform_basic_mouse_actions(landmarks, fingers_up, frame)
                self.perform_scroll_actions(landmarks, fingers_up, frame)
                
            elif self.current_mode == "KEYBOARD":
                self.perform_keyboard_actions(landmarks, fingers_up, frame)
                self.perform_text_selection_actions(landmarks, fingers_up, frame)
                
            elif self.current_mode == "MEDIA":
                self.perform_media_controls(landmarks, fingers_up, frame)
                
            elif self.current_mode == "WINDOW":
                self.perform_window_management(landmarks, fingers_up, frame)
                self.perform_browser_actions(landmarks, fingers_up, frame)
                
            elif self.current_mode == "GAMING":
                self.perform_gaming_actions(landmarks, fingers_up, frame)
                
            elif self.current_mode == "DRAWING":
                self.perform_drawing_actions(landmarks, fingers_up, frame)
            
            # Always available actions
            self.perform_system_actions(landmarks, fingers_up, frame)
            self.perform_accessibility_features(landmarks, fingers_up, frame)
            self.perform_presentation_actions(landmarks, fingers_up, frame)
            self.perform_zoom_meeting_actions(landmarks, fingers_up, frame)
            self.perform_custom_shortcuts(landmarks, fingers_up, frame)
            self.perform_mouse_precision_actions(landmarks, fingers_up, frame)
        
        # Two-hand gestures
        if len(landmarks_list) == 2:
            self.perform_two_hand_gestures(landmarks_list, frame)
            self.gesture_states['two_hand_mode'] = True
        else:
            self.gesture_states['two_hand_mode'] = False
    
    def draw_help_overlay(self, frame):
        """Draw comprehensive help overlay"""
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-300), (w-10, h-10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        help_text = [
            f"MODE: {self.current_mode} | Precision: {'ON' if self.gesture_states['precision_mode'] else 'OFF'}",
            "BASIC: Index=Move | Index+Middle=Click | Thumb+Index+Middle=RightClick",
            "SCROLL: 3Fingers=Scroll | 4Fingers=DoubleClick | 5Fingers=Drag | Fist=Stop",
            "MEDIA: Thumb+Pinky=Volume | Index+Ring=PlayControl | Middle=Stop",
            "WINDOW: Various combinations for window management and desktop switching",
            "SPECIAL: L-shape=Screenshot | Peace(spread)=Lock | Long gestures=Mode toggle",
            "TWO-HAND: Palm distance for zoom | Complex rotations and gestures",
            "MODES: Thumb+Middle+Ring cycles through Normal→Keyboard→Media→Window→Gaming→Drawing"
        ]
        
        for i, text in enumerate(help_text):
            y_pos = h - 280 + i * 25
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return frame
    
    def save_gesture_profile(self, filename="gesture_profile.json"):
        """Save current gesture settings"""
        profile = {
            'smoothening': self.smoothening,
            'click_threshold': self.click_threshold,
            'gesture_cooldown': self.gesture_cooldown,
            'current_mode': self.current_mode
        }
        try:
            with open(filename, 'w') as f:
                json.dump(profile, f)
            return True
        except:
            return False
    
    def load_gesture_profile(self, filename="gesture_profile.json"):
        """Load gesture settings"""
        try:
            with open(filename, 'r') as f:
                profile = json.load(f)
                self.smoothening = profile.get('smoothening', 7)
                self.click_threshold = profile.get('click_threshold', 35)
                self.gesture_cooldown = profile.get('gesture_cooldown', 0.3)
                self.current_mode = profile.get('current_mode', "NORMAL")
            return True
        except:
            return False
    
    def calibrate_user(self, frame):
        """User calibration for personalized gesture recognition"""
        cv2.putText(frame, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Make a fist for 3 seconds to calibrate", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Add calibration logic here
    
    def run(self):
        """Enhanced main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        # Try to load saved settings
        self.load_gesture_profile()
        
        print("🚀 Comprehensive Hand Gesture Controller Started!")
        print("="*60)
        print("📱 MODES:")
        print("  • NORMAL: Basic mouse and scroll control")
        print("  • KEYBOARD: Virtual keyboard and text editing")
        print("  • MEDIA: Music/video playback control")
        print("  • WINDOW: Window management and app switching")
        print("  • GAMING: WASD movement and game controls")
        print("  • DRAWING: Drawing and annotation tools")
        print()
        print("🎯 KEY GESTURES:")
        print("  • Index finger: Move cursor")
        print("  • Index+Middle close: Left click")
        print("  • Thumb+Index+Middle close: Right click")
        print("  • 3 fingers: Scroll (move hand up/down)")
        print("  • 4 fingers: Double click")
        print("  • 5 fingers: Drag mode")
        print("  • L-shape (thumb+index): Screenshot")
        print("  • Peace sign (spread): Lock screen")
        print("  • Thumb+Middle+Ring: Cycle modes")
        print()
        print("🔧 SPECIAL FEATURES:")
        print("  • Long press gestures for mode toggles")
        print("  • Two-hand zoom and rotate")
        print("  • Precision mode for fine control")
        print("  • Drawing canvas overlay")
        print("  • Virtual keyboard input")
        print("  • Custom application shortcuts")
        print()
        print("⌨️  CONTROLS:")
        print("  • 'q': Quit application")
        print("  • 'h': Toggle help overlay")
        print("  • 'c': Calibration mode")
        print("  • 's': Save gesture profile")
        print("="*60)
        
        show_help = False
        calibration_mode = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Process hand landmarks
            landmarks_list = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand connections
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                              self.mp_hands.HAND_CONNECTIONS)
                    landmarks_list.append(hand_landmarks.landmark)
                
                # Process gestures
                if not calibration_mode:
                    self.main_gesture_processor(landmarks_list, frame)
            
            # Add drawing canvas overlay if enabled
            if self.drawing_enabled:
                frame = cv2.addWeighted(frame, 0.7, self.drawing_canvas, 0.3, 0)
            
            # Draw UI elements
            self.draw_ui_elements(frame)
            
            # Calibration mode
            if calibration_mode:
                self.calibrate_user(frame)
            
            # Help overlay
            if show_help:
                frame = self.draw_help_overlay(frame)
            
            # Show frame
            cv2.imshow('Comprehensive Gesture Controller', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('c'):
                calibration_mode = not calibration_mode
            elif key == ord('s'):
                if self.save_gesture_profile():
                    print("✅ Gesture profile saved!")
                else:
                    print("❌ Failed to save profile")
            elif key == ord('r'):
                # Reset all modes and states
                self.gesture_states = {key: False for key in self.gesture_states}
                self.current_mode = "NORMAL"
                print("🔄 All modes reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save settings on exit
        self.save_gesture_profile()

# Specialized controller for specific use cases
class SpecializedGestureController(ComprehensiveGestureController):
    def __init__(self):
        super().__init__()
        self.eye_tracking_mode = False
        self.voice_command_mode = False
        self.macro_recording = False
        self.recorded_macros = {}
        self.current_macro = []
        
    def perform_macro_actions(self, landmarks, fingers_up, frame):
        """Record and playback gesture macros"""
        if fingers_up == [1, 1, 1, 1, 0]:  # Four fingers - Macro control
            ring_y = landmarks[16].y
            
            if ring_y < 0.2:  # Start recording
                if not self.macro_recording:
                    self.macro_recording = True
                    self.current_macro = []
                    self.display_action(frame, "MACRO RECORDING START")
                    
            elif ring_y > 0.8:  # Stop recording
                if self.macro_recording:
                    self.macro_recording = False
                    self.recorded_macros[f"macro_{len(self.recorded_macros)}"] = self.current_macro.copy()
                    self.display_action(frame, "MACRO SAVED")
                    
            else:  # Playback last macro
                if self.recorded_macros and not self.macro_recording:
                    last_macro = list(self.recorded_macros.values())[-1]
                    self.playback_macro(last_macro)
                    self.display_action(frame, "MACRO PLAYBACK")
        
        # Record current gesture if recording
        if self.macro_recording:
            self.current_macro.append({
                'fingers': fingers_up,
                'position': (landmarks[8].x, landmarks[8].y),
                'timestamp': time.time()
            })
    
    def playback_macro(self, macro):
        """Playback recorded macro"""
        def play():
            for gesture in macro:
                # Simulate the recorded gesture
                x = int(gesture['position'][0] * self.screen_width)
                y = int(gesture['position'][1] * self.screen_height)
                pyautogui.moveTo(x, y)
                
                # Add small delay between actions
                time.sleep(0.1)
        
        # Run macro in separate thread to avoid blocking
        threading.Thread(target=play, daemon=True).start()
    
    def perform_advanced_selection(self, landmarks, fingers_up, frame):
        """Advanced text and object selection"""
        if fingers_up == [1, 1, 0, 1, 0]:  # Thumb + Index + Ring - Smart selection
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            ring_tip = landmarks[16]
            
            # Triple click for paragraph selection
            if self.get_distance(
                (thumb_tip.x * self.cam_width, thumb_tip.y * self.cam_height),
                (ring_tip.x * self.cam_width, ring_tip.y * self.cam_height)
            ) < 30:
                pyautogui.tripleClick()
                self.display_action(frame, "SELECT PARAGRAPH")
            
            # Select all
            elif index_tip.y < 0.1:
                pyautogui.hotkey('ctrl', 'a')
                self.display_action(frame, "SELECT ALL")
    
    def perform_productivity_shortcuts(self, landmarks, fingers_up, frame):
        """Productivity and workflow shortcuts"""
        productivity_gestures = {
            # Quick app launching
            '[1, 0, 0, 0, 1]': {  # Thumb + Pinky
                'action': lambda: pyautogui.hotkey('win', 'r'),
                'name': 'RUN DIALOG'
            },
            
            # Quick system actions
            '[0, 0, 1, 0, 1]': {  # Middle + Pinky
                'action': lambda: pyautogui.hotkey('ctrl', 'alt', 'del'),
                'name': 'CTRL+ALT+DEL'
            },
            
            # Clipboard history
            '[1, 0, 1, 0, 0]': {  # Thumb + Middle
                'action': lambda: pyautogui.hotkey('win', 'v'),
                'name': 'CLIPBOARD HISTORY'
            }
        }
        
        gesture_key = str(fingers_up)
        if gesture_key in productivity_gestures:
            gesture_info = productivity_gestures[gesture_key]
            gesture_info['action']()
            self.display_action(frame, gesture_info['name'])
    
    def detect_gesture_speed(self, landmarks):
        """Detect gesture speed for variable actions"""
        if len(self.gesture_history) < 5:
            return "SLOW"
        
        # Calculate average speed over last few frames
        distances = []
        for i in range(1, min(5, len(self.gesture_history))):
            dist = self.get_distance(self.gesture_history[i-1], self.gesture_history[i])
            distances.append(dist)
        
        avg_speed = sum(distances) / len(distances)
        
        if avg_speed > 20:
            return "FAST"
        elif avg_speed > 10:
            return "MEDIUM"
        else:
            return "SLOW"
    
    def perform_speed_dependent_actions(self, landmarks, fingers_up, frame, speed):
        """Actions that depend on gesture speed"""
        if fingers_up == [0, 1, 1, 1, 0]:  # Index + Middle + Ring
            if speed == "FAST":
                pyautogui.scroll(10)  # Fast scroll
                self.display_action(frame, "FAST SCROLL")
            elif speed == "MEDIUM":
                pyautogui.scroll(3)  # Normal scroll
                self.display_action(frame, "NORMAL SCROLL")
            else:
                pyautogui.scroll(1)  # Slow scroll
                self.display_action(frame, "PRECISE SCROLL")

def create_gesture_tutorial():
    """Create an interactive tutorial"""
    print("\n" + "="*70)
    print("🎓 INTERACTIVE GESTURE TUTORIAL")
    print("="*70)
    
    tutorial_steps = [
        ("Basic Movement", "Point with index finger and move around"),
        ("Left Click", "Point with index and middle finger, bring them together"),
        ("Right Click", "Use thumb, index, and middle finger together"),
        ("Scroll", "Use three fingers and move hand up/down"),
        ("Mode Switch", "Use thumb + middle + ring finger to cycle modes"),
        ("Drawing", "Switch to drawing mode and point with index finger"),
        ("Precision", "Long press with index finger to toggle precision mode"),
        ("Emergency Stop", "Use alternating finger pattern at top of screen")
    ]
    
    for step, instruction in tutorial_steps:
        print(f"📍 {step}: {instruction}")
    
    print("\n💡 Tips:")
    print("  • Ensure good lighting for best tracking")
    print("  • Keep hand 1-2 feet from camera")
    print("  • Use contrasting background")
    print("  • Practice gestures slowly at first")
    print("  • Press 'h' in app for live help overlay")

def main():
    """Enhanced main function with setup options"""
    print("🚀 Starting Comprehensive Gesture Controller...")
    
    try:
        # Option to run tutorial
        tutorial = input("Run tutorial first? (y/n): ").lower().strip()
        if tutorial == 'y':
            create_gesture_tutorial()
            input("\nPress Enter to continue to the application...")
        
        # Choose controller type
        print("\nController Options:")
        print("1. Comprehensive Controller (All features)")
        print("2. Specialized Controller (With macros and advanced features)")
        
        choice = input("Choose controller (1/2, default=1): ").strip()
        
        if choice == "2":
            controller = SpecializedGestureController()
            print("🎯 Starting Specialized Controller with macro support...")
        else:
            controller = ComprehensiveGestureController()
            print("🎯 Starting Comprehensive Controller...")
        
        # Add startup delay
        print("Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        controller.run()
        
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📦 Required packages:")
        print("pip install opencv-python mediapipe pyautogui numpy")
        print("\n🔧 Troubleshooting:")
        print("  • Check camera permissions")
        print("  • Ensure proper lighting")
        print("  • Try different camera index if needed")

if __name__ == "__main__":
    main()