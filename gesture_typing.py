# gesture_drawing_board_final.py
import cv2
import numpy as np
import time
import math
from collections import deque

print("🎨 Starting HD Gesture Drawing Board...")

try:
    import mediapipe as mp
    print("✅ MediaPipe imported successfully!")
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except ImportError as e:
    print(f"⚠️  MediaPipe not available: {e}")
    print("Continuing with basic mode...")
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_draw = None

class GestureDrawingBoard:
    def __init__(self):
        print("Initializing HD Drawing Board...")
        
        # =============== CONFIGURABLE SETTINGS ===============
        # Finger gesture settings
        self.FINGERS_FOR_DRAW = 2      # ✌️ 2 fingers to write
        self.FINGERS_FOR_ERASE = 4     # 🖐️ 5+ fingers to erase
        self.FINGERS_FOR_MOVE = 1      # 👆 1 finger to move
        
        # Resolution settings
        self.SCREEN_WIDTH = 1920        # Full HD width
        self.SCREEN_HEIGHT = 1080       # Full HD height
        self.CAMERA_WIDTH = 1280        # Camera capture resolution
        self.CAMERA_HEIGHT = 720
        
        # Canvas settings
        self.CANVAS_WIDTH = 1920
        self.CANVAS_HEIGHT = 1080
        
        # Font settings
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_TITLE = 1.2
        self.FONT_SCALE_NORMAL = 0.8
        self.FONT_SCALE_SMALL = 0.6
        
        # Performance settings
        self.TARGET_FPS = 60
        self.FRAME_TIME = 1.0 / self.TARGET_FPS
        
        # Drawing settings
        self.DEFAULT_BRUSH_SIZE = 8
        self.DEFAULT_ERASER_SIZE = 40
        self.DEFAULT_COLOR = (0, 255, 0)  # Green
        # =============== END OF SETTINGS ===============
        
        # Initialize camera
        self.cap = None
        self.init_camera()
        
        # Drawing states
        self.drawing_mode = True
        self.current_color = self.DEFAULT_COLOR
        self.brush_size = self.DEFAULT_BRUSH_SIZE
        self.eraser_size = self.DEFAULT_ERASER_SIZE
        
        # Canvas
        self.canvas = None
        self.init_canvas(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        
        # Colors palette
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 255, 255),# White
            (0, 0, 0)       # Black
        ]
        
        # Hand tracking
        self.hand_pos = (0, 0)
        self.prev_hand_pos = (0, 0)
        self.is_drawing = False
        self.finger_count = 0
        
        # FIXED: Add point tracking for smooth lines
        self.last_draw_point = None  # Store last drawing point
        self.last_erase_point = None  # Store last erase point
        self.min_distance = 5  # Minimum distance between points to connect
        
        # UI
        self.ui_visible = True
        self.show_debug = True
        self.mouse_mode = False
        self.fullscreen = True
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        print("✅ HD Drawing Board Ready!")
        print(f"\n🎯 GESTURE CONTROLS:")
        print(f"   ✌️  {self.FINGERS_FOR_DRAW} Fingers - WRITE/DRAW")
        print(f"   🖐️  {self.FINGERS_FOR_ERASE}+ Fingers - ERASE")
        print(f"   👆  {self.FINGERS_FOR_MOVE} Finger - MOVE")
        print(f"\n📺 Resolution: {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        print(f"🎯 Target FPS: {self.TARGET_FPS}")
        print(f"\n⚙️  Controls:")
        print("   Space - Toggle draw/erase mode")
        print("   C - Change color")
        print("   S - Change brush size")
        print("   F - Toggle fullscreen")
        print("   ESC - Exit")
        
    def init_camera(self):
        """Initialize camera"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"Attempting to open camera (attempt {attempt + 1}/{max_attempts})...")
                self.cap = cv2.VideoCapture(0)
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
                
                # Try to read a frame
                ret, test_frame = self.cap.read()
                if ret:
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"✅ Camera opened successfully!")
                    print(f"   Resolution: {actual_width}x{actual_height}")
                    
                    # Adjust settings if camera doesn't support requested resolution
                    if actual_width < self.CAMERA_WIDTH:
                        print(f"⚠️  Camera doesn't support {self.CAMERA_WIDTH}p, using {actual_width}p")
                        self.CAMERA_WIDTH = actual_width
                        self.CAMERA_HEIGHT = actual_height
                    
                    return True
                else:
                    print("⚠️  Camera opened but couldn't read frame")
                    self.cap.release()
            except Exception as e:
                print(f"⚠️  Camera error: {e}")
                if self.cap:
                    self.cap.release()
            
            time.sleep(1)
        
        print("❌ Could not open camera. Using fallback mode.")
        self.mouse_mode = True
        return False
    
    def init_canvas(self, width, height):
        """Initialize canvas"""
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas[:] = (20, 20, 20)  # Dark gray background
    
    def init_hand_tracking(self):
        """Initialize hand tracking"""
        if not MEDIAPIPE_AVAILABLE:
            print("⚠️  MediaPipe not available")
            return None
        
        try:
            print("Initializing hand tracking...")
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            print("✅ Hand tracking initialized!")
            return hands
        except Exception as e:
            print(f"⚠️  Failed to initialize hand tracking: {e}")
            return None
    
    def put_text(self, frame, text, position, scale=0.8, thickness=2, color=(255, 255, 255)):
        """Draw text with specified font"""
        cv2.putText(frame, text, position, self.FONT_FACE, 
                   scale, color, thickness, cv2.LINE_AA)
    
    def detect_fingers_mediapipe(self, frame, hands):
        """Detect fingers using MediaPipe"""
        h, w = frame.shape[:2]
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 255), thickness=3, circle_radius=5),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 255, 0), thickness=2)
                )
                
                # Get index finger tip for cursor
                index_tip = hand_landmarks.landmark[8]
                hand_x = int(index_tip.x * w)
                hand_y = int(index_tip.y * h)
                
                # Count extended fingers
                finger_count = 0
                extended_fingers = []
                
                # Finger landmarks indices
                finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                finger_pips = [3, 6, 10, 14, 18]   # PIP joints
                
                for i in range(5):
                    tip = hand_landmarks.landmark[finger_tips[i]]
                    pip = hand_landmarks.landmark[finger_pips[i]]
                    
                    # Different logic for thumb
                    if i == 0:  # Thumb
                        thumb_tip = hand_landmarks.landmark[4]
                        thumb_ip = hand_landmarks.landmark[3]
                        dist_tip_ip = math.sqrt(
                            (thumb_tip.x - thumb_ip.x)**2 + 
                            (thumb_tip.y - thumb_ip.y)**2
                        )
                        is_extended = dist_tip_ip > 0.08
                    else:
                        # For other fingers, check if tip is above PIP
                        is_extended = tip.y < pip.y - 0.02
                    
                    if is_extended:
                        finger_count += 1
                        extended_fingers.append(i)
                
                return hand_x, hand_y, finger_count, extended_fingers
                
        except Exception as e:
            if self.show_debug:
                print(f"MediaPipe detection error: {e}")
        
        return None, None, 0, []
    
    def draw_on_canvas(self, x, y, mode="draw"):
        """Draw or erase on canvas with smooth lines"""
        if mode == "draw":
            # Draw a circle at current position
            cv2.circle(self.canvas, (x, y), self.brush_size, 
                      self.current_color, -1, lineType=cv2.LINE_AA)
            
            # FIXED: Connect to previous point for smooth lines
            if self.last_draw_point is not None:
                x1, y1 = self.last_draw_point
                x2, y2 = x, y
                
                # Calculate distance between points
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Only connect if distance is reasonable
                if distance < 100:  # Prevent connecting across large jumps
                    # Draw a thick line between points
                    cv2.line(self.canvas, (x1, y1), (x2, y2), 
                            self.current_color, self.brush_size * 2, 
                            lineType=cv2.LINE_AA)
            
            # Update last point
            self.last_draw_point = (x, y)
            
        elif mode == "erase":
            # Draw a circle at current position
            cv2.circle(self.canvas, (x, y), self.eraser_size, 
                      (20, 20, 20), -1, lineType=cv2.LINE_AA)
            
            # FIXED: Connect erase points for smooth erasing
            if self.last_erase_point is not None:
                x1, y1 = self.last_erase_point
                x2, y2 = x, y
                
                # Calculate distance between points
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Only connect if distance is reasonable
                if distance < 100:
                    # Draw a thick line between erase points
                    cv2.line(self.canvas, (x1, y1), (x2, y2), 
                            (20, 20, 20), self.eraser_size, 
                            lineType=cv2.LINE_AA)
            
            # Update last erase point
            self.last_erase_point = (x, y)
    
    def reset_last_points(self):
        """Reset last points when changing modes"""
        self.last_draw_point = None
        self.last_erase_point = None
    
    def draw_ui(self, frame, hand_x=None, hand_y=None, finger_count=0):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent UI panel
        ui_panel = np.zeros((h, 400, 3), dtype=np.uint8)
        frame[0:h, 0:400] = cv2.addWeighted(frame[0:h, 0:400], 0.7, ui_panel, 0.3, 0)
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (400, h), (100, 100, 100), 2)
        
        # Title
        self.put_text(frame, "GESTURE DRAWING BOARD", 
                     (20, 50), self.FONT_SCALE_TITLE, 3, (0, 255, 255))
        
        # Resolution info
        self.put_text(frame, f"RES: {w}x{h} @ {self.fps}FPS", 
                     (20, 90), 0.6, 1, (200, 200, 255))
        
        # Mode indicator
        mode_text = "WRITE MODE" if self.drawing_mode else "ERASE MODE"
        mode_color = (0, 255, 0) if self.drawing_mode else (255, 0, 0)
        self.put_text(frame, f"Mode: {mode_text}", 
                     (20, 130), self.FONT_SCALE_NORMAL, 2, mode_color)
        
        # Tracking mode
        tracking_mode = "MOUSE" if self.mouse_mode else "HAND GESTURES"
        self.put_text(frame, f"Input: {tracking_mode}", 
                     (20, 170), 0.7, 2, (255, 255, 0))
        
        # Finger count
        self.put_text(frame, f"Fingers: {finger_count}", 
                     (20, 210), 0.8, 2, (255, 200, 0))
        
        # Current color
        self.put_text(frame, "Current Color:", (20, 260), 0.7, 1, (200, 200, 200))
        cv2.circle(frame, (180, 255), 20, self.current_color, -1)
        cv2.circle(frame, (180, 255), 20, (255, 255, 255), 2)
        
        # Brush size
        self.put_text(frame, f"Brush: {self.brush_size}px", 
                     (20, 310), 0.7, 1, (200, 200, 200))
        cv2.circle(frame, (180, 305), self.brush_size, (255, 255, 255), -1)
        
        # Color palette
        self.put_text(frame, "Color Palette:", (20, 360), 0.7, 1, (200, 200, 200))
        for i, color in enumerate(self.colors):
            x_pos = 30 + (i % 4) * 45
            y_pos = 400 + (i // 4) * 45
            cv2.circle(frame, (x_pos, y_pos), 15, color, -1)
            cv2.circle(frame, (x_pos, y_pos), 15, (255, 255, 255), 2)
        
        # Gesture guide
        self.put_text(frame, "GESTURE GUIDE:", (20, 520), 0.8, 2, (0, 255, 255))
        guides = [
            f"✌️  {self.FINGERS_FOR_DRAW} Fingers = WRITE",
            f"🖐️  {self.FINGERS_FOR_ERASE}+ Fingers = ERASE",
            f"👆  {self.FINGERS_FOR_MOVE} Finger = MOVE"
        ]
        
        y_pos = 560
        for guide in guides:
            self.put_text(frame, guide, (30, y_pos), 0.6, 1, (200, 220, 255))
            y_pos += 35
        
        # Instructions
        self.put_text(frame, "CONTROLS:", (20, 680), 0.7, 2, (150, 200, 255))
        controls = [
            "Space - Toggle Mode",
            "C - Change Color",
            "S - Brush Size",
            "F - Fullscreen",
            "ESC - Exit"
        ]
        
        y_pos = 720
        for control in controls:
            self.put_text(frame, control, (30, y_pos), 0.5, 1, (180, 180, 255))
            y_pos += 30
        
        # Draw crosshair at hand position
        if hand_x and hand_y and not self.mouse_mode:
            crosshair_size = 40
            cv2.line(frame, (hand_x-crosshair_size, hand_y), 
                    (hand_x+crosshair_size, hand_y), (0, 255, 255), 2)
            cv2.line(frame, (hand_x, hand_y-crosshair_size), 
                    (hand_x, hand_y+crosshair_size), (0, 255, 255), 2)
            cv2.circle(frame, (hand_x, hand_y), 25, (0, 255, 255), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def handle_mouse(self, event, x, y, flags, param):
        """Mouse callback for drawing"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.hand_pos = (x, y)
            if flags & cv2.EVENT_FLAG_LBUTTON:
                mode = "draw" if self.drawing_mode else "erase"
                self.draw_on_canvas(x, y, mode)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.reset_last_points()  # Reset when starting new stroke
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.reset_last_points()  # Reset when releasing mouse
    
    def run(self):
        """Main drawing loop"""
        print("\n" + "="*70)
        print("🎨 STARTING DRAWING SESSION")
        print("="*70)
        
        # Initialize hand tracking
        hands = self.init_hand_tracking()
        
        # Create window
        window_name = "Gesture Drawing Board - Press ESC to Exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        if self.mouse_mode:
            cv2.setMouseCallback(window_name, self.handle_mouse)
        
        print("🖥️  Window created")
        print("🎮 Ready for gestures!")
        print("="*70)
        
        last_finger_count = 0
        
        while True:
            try:
                frame_start_time = time.time()
                
                # Read frame from camera
                if not self.mouse_mode and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("⚠️  Could not read frame from camera")
                        self.mouse_mode = True
                        continue
                else:
                    # Create blank frame for mouse mode
                    frame = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
                    frame[:] = (30, 30, 30)
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Update screen dimensions
                self.SCREEN_WIDTH = w
                self.SCREEN_HEIGHT = h
                
                # Resize canvas if needed
                if self.canvas.shape[:2] != (h, w):
                    self.init_canvas(w, h)
                
                # Hand detection
                hand_x, hand_y = None, None
                finger_count = 0
                
                if not self.mouse_mode and hands:
                    hand_x, hand_y, finger_count, _ = self.detect_fingers_mediapipe(frame, hands)
                
                # FIXED: Detect gesture changes and reset points
                if finger_count != last_finger_count:
                    self.reset_last_points()
                    last_finger_count = finger_count
                
                # Handle gestures based on finger count
                if hand_x and hand_y and not self.mouse_mode:
                    # Update hand position
                    self.hand_pos = (hand_x, hand_y)
                    
                    # GESTURE LOGIC:
                    if finger_count == self.FINGERS_FOR_DRAW:  # ✌️ 2 fingers to write
                        if self.drawing_mode:
                            self.draw_on_canvas(hand_x, hand_y, "draw")
                    
                    elif finger_count >= self.FINGERS_FOR_ERASE:  # 🖐️ 5+ fingers to erase
                        self.draw_on_canvas(hand_x, hand_y, "erase")
                    
                    else:
                        # Reset points when not drawing/erasing
                        self.reset_last_points()
                
                # Blend canvas with camera feed
                canvas_display = cv2.resize(self.canvas, (w, h))
                frame = cv2.addWeighted(frame, 0.4, canvas_display, 0.6, 0)
                
                # Draw UI
                if self.ui_visible:
                    frame = self.draw_ui(frame, hand_x, hand_y, finger_count)
                
                # Update and display FPS
                self.update_fps()
                if self.show_debug:
                    self.put_text(frame, f"FPS: {self.fps}", 
                                 (w - 150, 40), 0.8, 2, (255, 255, 0))
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n🛑 ESC pressed - Exiting...")
                    break
                elif key == ord(' '):
                    self.drawing_mode = not self.drawing_mode
                    mode_text = "WRITE" if self.drawing_mode else "ERASE"
                    print(f"✍️  Mode changed to: {mode_text}")
                    self.reset_last_points()  # Reset when changing modes
                elif key == ord('c'):
                    current_idx = self.colors.index(self.current_color) if self.current_color in self.colors else 0
                    self.current_color = self.colors[(current_idx + 1) % len(self.colors)]
                    color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "White", "Black"]
                    color_name = color_names[current_idx]
                    print(f"🎨 Color changed to: {color_name}")
                elif key == ord('s'):
                    sizes = [5, 8, 12, 20, 30, 50]
                    current_idx = sizes.index(self.brush_size) if self.brush_size in sizes else 0
                    self.brush_size = sizes[(current_idx + 1) % len(sizes)]
                    print(f"🖌️  Brush size: {self.brush_size}px")
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("📺 Switched to FULLSCREEN")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, 1280, 720)
                        print("📺 Switched to WINDOWED mode")
                elif key == ord('p'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"drawing_{timestamp}.png"
                    cv2.imwrite(filename, self.canvas)
                    print(f"💾 Drawing saved as: {filename}")
                elif key == ord('r'):
                    # Clear canvas
                    self.init_canvas(w, h)
                    self.reset_last_points()
                    print("🧹 Canvas cleared!")
                
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                continue
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        
        print("\n" + "="*70)
        print("🎨 DRAWING SESSION COMPLETE")
        print("="*70)
        print(f"Final Resolution: {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        print(f"Average FPS: {self.fps}")
        print("Thank you for using Gesture Drawing Board!")
        print("="*70)

def main():
    """Main entry point"""
    print("="*70)
    print("🎨 GESTURE DRAWING BOARD")
    print("="*70)
    print("Features:")
    print("  • Full HD 1080p Resolution")
    print("  • 60 FPS Performance")
    print("  • ✌️  2 Fingers = WRITE")
    print("  • 🖐️  5+ Fingers = ERASE")
    print("  • 👆  1 Finger = MOVE")
    print("  • ✅ SMOOTH LINE DRAWING (Fixed)")
    print("="*70)
    
    try:
        board = GestureDrawingBoard()
        board.run()
    except Exception as e:
        print(f"\n💀 UNEXPECTED ERROR: {e}")
        print("Please check your camera and try again.")

if __name__ == "__main__":
    main()