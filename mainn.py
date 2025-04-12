from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color, Ellipse, Rectangle, Triangle, Callback
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.core.text import Label as CoreLabel
from kivy.graphics.texture import Texture
import math
import random
import cv2
import numpy as np
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate
import smbus
import time

# Try to import picamera2 for Raspberry Pi 5 Module 3 camera
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("Picamera2 module available")
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Picamera2 module not available")

# MPU6050 constants and setup
MPU6050_ADDR = 0x68
MPU6050_SMPLRT_DIV = 0x19
MPU6050_CONFIG = 0x1A
MPU6050_GYRO_CONFIG = 0x1B
MPU6050_ACCEL_CONFIG = 0x1C
MPU6050_INT_ENABLE = 0x38
MPU6050_PWR_MGMT_1 = 0x6B
MPU6050_TEMP_OUT_H = 0x41
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_GYRO_XOUT_H = 0x43


class MPU6050:
    def __init__(self, address=MPU6050_ADDR, bus=1):
        self.address = address
        self.bus = smbus.SMBus(bus)
        
        # Wake up the MPU6050
        self.bus.write_byte_data(self.address, MPU6050_PWR_MGMT_1, 0)
        
        # Configure the accelerometer (+/-8g)
        self.bus.write_byte_data(self.address, MPU6050_ACCEL_CONFIG, 0x10)
        
        # Configure the gyroscope (500deg/s)
        self.bus.write_byte_data(self.address, MPU6050_GYRO_CONFIG, 0x08)
        
        # Set sample rate to 100 Hz
        self.bus.write_byte_data(self.address, MPU6050_SMPLRT_DIV, 0x09)
        
        # Calibration values
        self.accel_scale = 4096.0  # For +/-8g
        self.gyro_scale = 65.5     # For 500deg/s
        
        # Initialize variables
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        
        # Time for integration
        self.last_time = time.time()
        
        # Complementary filter coefficient
        self.alpha = 0.98
        
        print("MPU6050 initialized")
        
    def read_word(self, reg):
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def read_accel_gyro(self):
        accel_x = self.read_word(MPU6050_ACCEL_XOUT_H) / self.accel_scale
        accel_y = self.read_word(MPU6050_ACCEL_XOUT_H + 2) / self.accel_scale
        accel_z = self.read_word(MPU6050_ACCEL_XOUT_H + 4) / self.accel_scale
        
        gyro_x = self.read_word(MPU6050_GYRO_XOUT_H) / self.gyro_scale
        gyro_y = self.read_word(MPU6050_GYRO_XOUT_H + 2) / self.gyro_scale
        gyro_z = self.read_word(MPU6050_GYRO_XOUT_H + 4) / self.gyro_scale
        
        return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

    def update_orientation(self):
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = self.read_accel_gyro()
        
        # Calculate pitch and roll from accelerometer (inverting signs)
        accel_pitch = -math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180 / math.pi
        accel_roll = -math.atan2(-accel_x, accel_z) * 180 / math.pi
        
        # Time difference
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Integrate gyroscope data (with inverted signs)
        self.pitch -= gyro_x * dt
        self.roll -= gyro_y * dt
        self.yaw -= gyro_z * dt
        
        # Apply complementary filter
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch
        self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll
        
        # Normalize yaw to keep it between 0-360
        self.yaw = self.yaw % 360
        
        return self.pitch, self.roll, self.yaw


class StarkHUDWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heading = 0
        self.altitude = 100
        self.speed = 85
        self.power = 87
        self.pitch = 0   # Attitude pitch in degrees
        self.roll = 0    # Attitude roll in degrees
        self.yaw = 0     # Attitude yaw in degrees 
        self.target_locked = False
        self.scanning = True
        self.system_status = "ALL SYSTEMS NOMINAL"
        self.scan_angle = 0
        self.data_points = []
        for _ in range(30):  # Generate some random data for visualization
            self.data_points.append(random.uniform(0.2, 0.8))
            
        # Initialize camera variables
        self.frame_width = 640
        self.frame_height = 480
        self.capture = None
        self.picam = None
        self.using_picamera = False
        
        # Initialize MPU6050 sensor
        self.mpu6050 = MPU6050()
        
        # On Raspberry Pi, try Pi camera first, otherwise try webcam
        if PICAMERA_AVAILABLE:
            if self.try_picamera():
                print("Using Pi camera as primary camera")
            else:
                self.try_webcam()
        else:
            self.try_webcam()
        
        # Create texture for camera frame with correct dimensions
        self.texture = Texture.create(size=(self.frame_width, self.frame_height), colorfmt='rgb')
        
        Clock.schedule_interval(self.update, 1/30)  # Update at 30 FPS
    
    def try_webcam(self):
        """Try to initialize standard webcam"""
        try:
            self.capture = cv2.VideoCapture(0)
            
            # Check if webcam opened successfully
            if not self.capture.isOpened():
                self.system_status = "WEBCAM ERROR: CANNOT ACCESS CAMERA"
                print("Error: Could not open webcam")
                return False
            
            # Get webcam resolution
            ret, frame = self.capture.read()
            if ret:
                self.frame_height, self.frame_width = frame.shape[:2]
                self.system_status = "STANDARD WEBCAM CONNECTED"
                print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
                return True
            else:
                self.system_status = "WEBCAM ERROR: CANNOT GET FRAME"
                print("Error: Could not get frame from webcam")
                self.capture.release()
                self.capture = None
                return False
        except Exception as e:
            self.system_status = f"WEBCAM ERROR: {str(e)}"
            print(f"Error initializing webcam: {e}")
            return False
        
    def try_picamera(self):
        """Initialize Raspberry Pi 5 Module 3 camera"""
        if not PICAMERA_AVAILABLE:
            self.system_status = "PI CAMERA NOT AVAILABLE"
            return False
            
        try:
            # Initialize the camera with a simpler approach
            self.picam = Picamera2()
            
            # Configure camera with preview config
            config = self.picam.create_preview_configuration(
                main={"size": (self.frame_width, self.frame_height), "format": "RGB888"}
            )
            self.picam.configure(config)
            
            # Start the camera
            self.picam.start()
            
            # Wait a moment for camera to initialize
            import time
            time.sleep(1.0)
            
            # Test if we can get a frame
            test_frame = self.picam.capture_array()
            
            if test_frame is not None and len(test_frame.shape) == 3:
                self.frame_height, self.frame_width = test_frame.shape[:2]
                self.using_picamera = True
                self.system_status = f"PI CAMERA ACTIVE: {self.frame_width}x{self.frame_height}"
                print(f"Successfully initialized Pi Camera: {self.frame_width}x{self.frame_height}")
                return True
            else:
                self.system_status = "PI CAMERA ERROR: INVALID FRAME"
                print(f"Invalid frame from Pi Camera: {test_frame}")
                return False
        except Exception as e:
            self.system_status = f"PI CAMERA ERROR: {str(e)}"
            print(f"Error initializing Pi Camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def __del__(self):
        # Release camera resources when app closes
        self.release_camera()
    
    def release_camera(self):
        if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
            self.capture.release()
            print("OpenCV Camera released")
            
        if hasattr(self, 'picam') and self.picam and self.using_picamera:
            try:
                self.picam.close()
                print("Pi Camera released")
            except:
                pass
        
    def update(self, dt):
        # Update animation values
        self.scan_angle = (self.scan_angle + 5) % 360
        self.heading = (self.heading + 0.5) % 360
        
        # Update MPU6050 orientation
        self.pitch, self.roll, self.yaw = self.mpu6050.update_orientation()
        
        # Get camera frame
        frame = self.get_camera_frame()
        
        # If we have a frame, update the texture
        if frame is not None:
            # Create or update texture
            if (self.texture.width != frame.shape[1] or 
                self.texture.height != frame.shape[0]):
                self.texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]), 
                    colorfmt='rgb'
                )
            
            # Convert frame to buffer and update texture
            buf = frame.tobytes()
            self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        
        # Redraw the HUD
        self.canvas.clear()
        self.draw_elements()

    def get_camera_frame(self):
        """Get a frame from the active camera (Pi Camera or webcam)"""
        if self.using_picamera and self.picam:
            try:
                # Get frame from Pi Camera (already in RGB format from config)
                frame = self.picam.capture_array()
                
                # No debug text - removed
                return frame
                    
            except Exception as e:
                print(f"Error capturing Pi Camera frame: {e}")
                return None
                    
        elif self.capture and self.capture.isOpened():
            try:
                # Get frame from OpenCV camera
                ret, frame = self.capture.read()
                
                if ret:
                    # Convert BGR to RGB for Kivy
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # No debug text - removed
                    return frame
                else:
                    return None
            except Exception as e:
                print(f"Error capturing webcam frame: {e}")
                return None
        
        return None
        
    def draw_elements(self):
        center_x = self.width / 2
        center_y = self.height / 2
        
        with self.canvas:
            # Draw camera feed as background first (if available)
            if hasattr(self, 'texture'):
                Color(1, 1, 1, 1)  # Full opacity for camera feed
                Rectangle(texture=self.texture, pos=(0, 0), size=(self.width, self.height))
                
            # Background elements - hexagonal grid pattern
            Color(0, 0.7, 0.9, 0.1)  # Iron Man blue with low opacity
            self.draw_hex_grid(20, 20, center_x, center_y)
            
            # Main targeting reticle
            self.draw_targeting_reticle(center_x, center_y)
            
            # Top arc with heading
            self.draw_heading_arc(center_x, self.height - 50)
            
            # Bottom status display
            self.draw_status_bar(center_x, 50)
            
            # Left side power display
            self.draw_power_indicator(60, center_y)
            
            # Right side altitude display
            self.draw_altitude_indicator(self.width - 60, center_y)
            
            # Pitch and Roll attitude indicator instead of scanning
            self.draw_attitude_indicator(center_x, center_y)
            
            # Data visualization on edges
            self.draw_data_visualization()

    def draw_hex_grid(self, rows, cols, center_x, center_y):
        hex_size = 30
        width = hex_size * cols * 1.5
        height = hex_size * rows * 0.866 * 2
        
        start_x = center_x - width/2
        start_y = center_y - height/2
        
        for row in range(rows):
            for col in range(cols):
                # Stagger every other row
                offset_x = hex_size * 0.75 if row % 2 else 0
                
                x = start_x + col * hex_size * 1.5 + offset_x
                y = start_y + row * hex_size * 1.732
                
                # Only draw if in visible area and not in center (to keep center cleaner)
                dist_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if 0 < x < self.width and 0 < y < self.height and dist_from_center > 100:
                    self.draw_hexagon(x, y, hex_size/3)

    def draw_hexagon(self, x, y, size):
        points = []
        for i in range(6):
            angle = math.radians(60 * i + 30)
            points.extend([
                x + size * math.cos(angle),
                y + size * math.sin(angle)
            ])
        Line(points=points, width=1, close=True)

    def draw_targeting_reticle(self, x, y):
        # Main targeting reticle - Iron Man style circular elements
        reticle_size = 120
        
        # Outer circle
        Color(0, 0.7, 0.9, 0.8)  # Iron Man blue
        Line(circle=(x, y, reticle_size), width=1.5)
        
        # Inner rotating elements
        Color(0, 0.7, 0.9, 0.6)
        
        # Inner circle
        Line(circle=(x, y, reticle_size * 0.7), width=1)
        
        # Dynamic rotating elements
        PushMatrix()
        Rotate(origin=(x, y), angle=self.scan_angle)
        
        # Tick marks around inner circle
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            inner_r = reticle_size * 0.7
            outer_r = reticle_size * 0.8
            x1 = x + inner_r * math.cos(rad)
            y1 = y + inner_r * math.sin(rad)
            x2 = x + outer_r * math.cos(rad)
            y2 = y + outer_r * math.sin(rad)
            Line(points=[x1, y1, x2, y2], width=1)
        
        # Crosshairs
        Line(points=[x - reticle_size*0.5, y, x - reticle_size*0.2, y], width=1)
        Line(points=[x + reticle_size*0.2, y, x + reticle_size*0.5, y], width=1)
        Line(points=[x, y - reticle_size*0.5, x, y - reticle_size*0.2], width=1)
        Line(points=[x, y + reticle_size*0.2, x, y + reticle_size*0.5], width=1)
        PopMatrix()
        
        # Central dynamic element
        Color(1, 1, 1, 0.9)
        reticle_inner = 15
        Line(circle=(x, y, reticle_inner), width=1)
        
        # Small triangles at cardinal points
        triangle_size = 5
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle)
            tx = x + reticle_inner * math.cos(rad)
            ty = y + reticle_inner * math.sin(rad)
            
            # Triangle points
            p1x = tx - triangle_size/2
            p1y = ty - triangle_size/2
            p2x = tx + triangle_size/2
            p2y = ty - triangle_size/2
            p3x = tx
            p3y = ty + triangle_size/2
            
            # Rotate points based on angle
            if angle == 0:  # Right
                p1x, p1y = tx, ty - triangle_size/2
                p2x, p2y = tx, ty + triangle_size/2
                p3x, p3y = tx + triangle_size, ty
            elif angle == 90:  # Top
                p1x, p1y = tx - triangle_size/2, ty
                p2x, p2y = tx + triangle_size/2, ty
                p3x, p3y = tx, ty + triangle_size
            elif angle == 180:  # Left
                p1x, p1y = tx, ty - triangle_size/2
                p2x, p2y = tx, ty + triangle_size/2
                p3x, p3y = tx - triangle_size, ty
            else:  # Bottom
                p1x, p1y = tx - triangle_size/2, ty
                p2x, p2y = tx + triangle_size/2, ty
                p3x, p3y = tx, ty - triangle_size
            
            Triangle(points=[p1x, p1y, p2x, p2y, p3x, p3y])

    def draw_attitude_indicator(self, x, y):
        # Artificial horizon / attitude indicator
        attitude_size = 180  # Size of the attitude indicator
        
        # Outer frame
        Color(0, 0.7, 0.9, 0.7)
        Line(circle=(x, y, attitude_size), width=1.5)
        
        # Save state before rotation
        PushMatrix()
        # Apply roll rotation
        Rotate(origin=(x, y), angle=self.roll)
        
        # Calculate pitch offset (pixels per degree)
        pixels_per_degree = 2.5
        pitch_offset = self.pitch * pixels_per_degree
        
        # Horizon line
        Color(1, 1, 1, 0.8)
        Line(points=[x - attitude_size, y - pitch_offset, 
                     x + attitude_size, y - pitch_offset], width=2)
        
        # Pitch ladder (lines above and below horizon)
        for degrees in range(-90, 91, 10):
            if degrees == 0:  # Skip 0 degrees (horizon already drawn)
                continue
                
            # Calculate y position based on pitch
            ladder_y = y - pitch_offset + degrees * pixels_per_degree
            
            # Only draw if in visible range
            if y - attitude_size <= ladder_y <= y + attitude_size:
                # Determine line length based on angle
                line_length = attitude_size * 0.5 if degrees % 30 == 0 else attitude_size * 0.2
                
                Line(points=[x - line_length/2, ladder_y, 
                             x + line_length/2, ladder_y], width=1)
                             
                # Add degree numbers for major angles
                if degrees % 30 == 0:
                    degree_label = CoreLabel(text=f"{abs(degrees)}°", font_size=10)
                    degree_label.refresh()
                    texture = degree_label.texture
                    
                    # Position text at the end of the line
                    text_x = x - line_length/2 - texture.width - 5 if degrees > 0 else x + line_length/2 + 5
                    Rectangle(pos=(text_x, ladder_y - texture.height/2),
                              size=texture.size, texture=texture)
        
        PopMatrix()
        
        # Draw fixed reference marker (aircraft symbol)
        Color(1, 0.8, 0.0, 0.9)  # Restored bright gold/yellow
        
        # Central dot
        Line(circle=(x, y, 2), width=2)
        
        # Aircraft wings
        wing_width = 25
        Line(points=[x - wing_width, y, x - 10, y], width=2)
        Line(points=[x + 10, y, x + wing_width, y], width=2)
        
        # Vertical stabilizer
        Line(points=[x, y, x, y - 10], width=2)
        
        # Attitude values display
        value_x = x + attitude_size + 15
        value_y = y + 40
        spacing = 20
        
        Color(0, 0.7, 0.9, 0.9)
        
        # Pitch value
        pitch_text = f"PITCH: {self.pitch:.1f}°"
        label = CoreLabel(text=pitch_text, font_size=12)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(value_x, value_y), size=texture.size, texture=texture)
        
        # Roll value
        roll_text = f"ROLL: {self.roll:.1f}°"
        label = CoreLabel(text=roll_text, font_size=12)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(value_x, value_y - spacing), size=texture.size, texture=texture)
        
        # Yaw value
        yaw_text = f"YAW: {self.yaw:.1f}°"
        label = CoreLabel(text=yaw_text, font_size=12)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(value_x, value_y - spacing*2), size=texture.size, texture=texture)
        
        # Draw roll indicator at the top of the attitude indicator
        roll_indicator_radius = attitude_size + 15
        
        # Draw roll scale arc
        Color(0, 0.7, 0.9, 0.5)
        
        # Draw roll scale tick marks
        for roll_angle in range(-60, 61, 10):
            angle_rad = math.radians(roll_angle - 90)  # -90 to rotate to top
            tick_x = x + roll_indicator_radius * math.cos(angle_rad)
            tick_y = y + roll_indicator_radius * math.sin(angle_rad)
            
            # Longer ticks for major angles
            tick_length = 10 if roll_angle % 30 == 0 else 5
            
            inner_x = x + (roll_indicator_radius - tick_length) * math.cos(angle_rad)
            inner_y = y + (roll_indicator_radius - tick_length) * math.sin(angle_rad)
            
            Line(points=[inner_x, inner_y, tick_x, tick_y], width=1)
            
            # Add labels for major tick marks
            if roll_angle % 30 == 0 and roll_angle != 0:
                label = CoreLabel(text=f"{abs(roll_angle)}", font_size=10)
                label.refresh()
                texture = label.texture
                label_x = x + (roll_indicator_radius + 5) * math.cos(angle_rad) - texture.width/2
                label_y = y + (roll_indicator_radius + 5) * math.sin(angle_rad) - texture.height/2
                Rectangle(pos=(label_x, label_y), size=texture.size, texture=texture)
        
        # Draw the roll indicator arrow
        Color(1, 0.8, 0.0, 0.9)  # Restored bright gold/yellow
        roll_rad = math.radians(self.roll - 90)  # -90 to rotate to top
        arrow_x = x + roll_indicator_radius * math.cos(roll_rad)
        arrow_y = y + roll_indicator_radius * math.sin(roll_rad)
        
        # Arrow shape
        triangle_size = 8
        triangle_points = [
            arrow_x, arrow_y,
            arrow_x - triangle_size/2, arrow_y - triangle_size,
            arrow_x + triangle_size/2, arrow_y - triangle_size
        ]
        Triangle(points=triangle_points)

    def draw_heading_arc(self, x, y):
        arc_width = 400
        arc_height = 60
        
        # Draw arc background
        Color(0, 0.7, 0.9, 0.2)
        Rectangle(pos=(x - arc_width/2, y - arc_height/2),
                  size=(arc_width, arc_height))
        
        # Draw heading markers
        Color(0, 0.7, 0.9, 0.7)
        for deg in range(0, 360, 10):
            rel_pos = ((deg - self.heading) % 360) / 360
            if 0.1 <= rel_pos <= 0.9:  # Only show portion of compass
                marker_x = x - arc_width/2 + rel_pos * arc_width
                marker_height = arc_height/4 if deg % 30 == 0 else arc_height/8
                Line(points=[marker_x, y - marker_height/2, marker_x, y + marker_height/2], width=1)
                
                if deg % 30 == 0:
                    # Add degree text
                    label = CoreLabel(text=f"{deg}°", font_size=10)
                    label.refresh()
                    texture = label.texture
                    Rectangle(pos=(marker_x - texture.width/2, y + marker_height), 
                              size=texture.size, texture=texture)
        
        # Draw center heading indicator (triangle)
        Color(1, 1, 1, 0.9)
        triangle_size = 10
        triangle_points = [
            x, y + arc_height/2 + triangle_size,  # Top
            x - triangle_size/2, y + arc_height/2,  # Bottom left
            x + triangle_size/2, y + arc_height/2   # Bottom right
        ]
        Line(points=triangle_points, width=1.5, close=True)
        
        # Current heading text
        heading_text = f"HDG {int(self.heading)}°"
        label = CoreLabel(text=heading_text, font_size=16)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(x - texture.width/2, y - 30), size=texture.size, texture=texture)

    def draw_status_bar(self, x, y):
        bar_width = 500
        bar_height = 30
        
        # Status bar background
        Color(0, 0.7, 0.9, 0.2)
        Rectangle(pos=(x - bar_width/2, y - bar_height/2),
                  size=(bar_width, bar_height))
        
        # Status text
        Color(1, 1, 1, 0.9)
        label = CoreLabel(text=self.system_status, font_size=14)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(x - texture.width/2, y - texture.height/2), 
                  size=texture.size, texture=texture)
        
        # Speed indicator on the left
        speed_x = x - bar_width/2 - 80
        speed_text = f"SPD: {self.speed} KM/H"
        speed_label = CoreLabel(text=speed_text, font_size=14)
        speed_label.refresh()
        texture = speed_label.texture
        Rectangle(pos=(speed_x - texture.width/2, y - texture.height/2), 
                  size=texture.size, texture=texture)

    def draw_power_indicator(self, x, y):
        indicator_height = 300
        indicator_width = 40
        
        # Background
        Color(0, 0.7, 0.9, 0.2)
        Rectangle(pos=(x - indicator_width/2, y - indicator_height/2),
                  size=(indicator_width, indicator_height))
        
        # Power level
        Color(0, 0.7, 0.9, 0.6)
        power_height = (self.power / 100) * indicator_height
        Rectangle(pos=(x - indicator_width/2, y - indicator_height/2),
                  size=(indicator_width, power_height))
        
        # Ticks
        Color(1, 1, 1, 0.7)
        for i in range(11):  # 0% to 100% in steps of 10%
            tick_y = y - indicator_height/2 + (i/10) * indicator_height
            tick_width = indicator_width if i % 5 == 0 else indicator_width * 0.7
            Line(points=[x - tick_width/2, tick_y, x + tick_width/2, tick_y], width=1)
            
            if i % 2 == 0:
                # Add percentage text
                label = CoreLabel(text=f"{i*10}%", font_size=10)
                label.refresh()
                texture = label.texture
                Rectangle(pos=(x - indicator_width/2 - texture.width - 5, tick_y - texture.height/2), 
                          size=texture.size, texture=texture)
        
        # Power text at top
        label = CoreLabel(text="POWER", font_size=12)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(x - texture.width/2, y + indicator_height/2 + 5), 
                  size=texture.size, texture=texture)

    def draw_altitude_indicator(self, x, y):
        indicator_height = 300
        indicator_width = 40
        
        # Background
        Color(0, 0.7, 0.9, 0.2)
        Rectangle(pos=(x - indicator_width/2, y - indicator_height/2),
                  size=(indicator_width, indicator_height))
        
        # Altitude ticks
        Color(1, 1, 1, 0.7)
        for i in range(11):  # 0m to 200m in steps of 20m
            tick_y = y - indicator_height/2 + (i/10) * indicator_height
            tick_width = indicator_width if i % 5 == 0 else indicator_width * 0.7
            Line(points=[x - tick_width/2, tick_y, x + tick_width/2, tick_y], width=1)
            
            if i % 2 == 0:
                # Add altitude text
                alt_value = i * 20  # 0, 40, 80, etc.
                label = CoreLabel(text=f"{alt_value}m", font_size=10)
                label.refresh()
                texture = label.texture
                Rectangle(pos=(x + indicator_width/2 + 5, tick_y - texture.height/2), 
                          size=texture.size, texture=texture)
        
        # Current altitude marker - restored to orange
        marker_y = y - indicator_height/2 + (self.altitude / 200) * indicator_height
        Color(1, 0.5, 0, 0.9)  # Restored orange-yellow
        triangle_size = 8
        triangle_points = [
            x - indicator_width/2 - triangle_size, marker_y,  # Left
            x - indicator_width/2, marker_y + triangle_size/2,  # Top
            x - indicator_width/2, marker_y - triangle_size/2   # Bottom
        ]
        Triangle(points=triangle_points)
        
        # Altitude text at top
        Color(1, 1, 1, 0.9)
        label = CoreLabel(text="ALTITUDE", font_size=12)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(x - texture.width/2, y + indicator_height/2 + 5), 
                  size=texture.size, texture=texture)
        
        # Current altitude
        alt_text = f"{self.altitude}m"
        label = CoreLabel(text=alt_text, font_size=14)
        label.refresh()
        texture = label.texture
        Rectangle(pos=(x - texture.width/2, y - indicator_height/2 - 25), 
                  size=texture.size, texture=texture)

    def draw_data_visualization(self):
        # Data visualization bars along the edges
        Color(0, 0.7, 0.9, 0.4)
        
        # Left edge
        bar_width = 5
        bar_spacing = 10
        num_bars = min(len(self.data_points), 20)
        
        for i in range(num_bars):
            height = self.data_points[i] * 50
            x = 10 + i * (bar_width + bar_spacing)
            y = 120
            Rectangle(pos=(x, y), size=(bar_width, height))
            
        # Right edge
        for i in range(num_bars):
            height = self.data_points[(i+10) % len(self.data_points)] * 50
            x = self.width - 10 - (i+1) * (bar_width + bar_spacing)
            y = 120
            Rectangle(pos=(x, y), size=(bar_width, height))

class StarkHUDApp(App):
    def build(self):
        root = FloatLayout()
        hud = StarkHUDWidget()
        root.add_widget(hud)
        return root

if __name__ == '__main__':
    StarkHUDApp().run()