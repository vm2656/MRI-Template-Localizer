import cv2
import numpy as np
import pandas as pd
import pyzed.sl as sl
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque
from typing import Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('needle_tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

class StereoNeedleTracker:
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.zed = sl.Camera()
        self.runtime_params = None
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.point_cloud = sl.Mat()
        self.running = False

        # Track multiple points
        self.current_coordinates = [None] * self.config['num_points']
        # Separate history for each point
        self.coord_histories = [deque(maxlen=5) for _ in range(self.config['num_points'])]

        self._setup_data_storage()

    def smooth_coordinates(self, coords: np.ndarray, point_index: int) -> np.ndarray:
        """
        Apply smoothing to coordinates using moving average.
        Args:
            coords: Current coordinates [x, y, z]
            point_index: Index of the point being smoothed
        Returns:
            Smoothed coordinates
        """
        try:
            # Add current coordinates to history for this point
            self.coord_histories[point_index].append(coords)

            # Apply smoothing if we have enough history
            if len(self.coord_histories[point_index]) >= 3:
                coords_array = np.array(self.coord_histories[point_index])
                median_coords = np.median(coords_array, axis=0)

                valid_coords = []
                for c in coords_array:
                    if np.all(np.abs(c - median_coords) < 0.2 * np.abs(median_coords)):
                        valid_coords.append(c)

                if valid_coords:
                    smoothed = np.mean(valid_coords, axis=0)
                    logger.debug(f"Smoothed coordinates for point {point_index+1} from {len(valid_coords)} samples")
                    return smoothed

            return coords

        except Exception as e:
            logger.error(f"Error in coordinate smoothing: {str(e)}")
            return coords

    def _get_default_config(self) -> Dict:
        """Return default configuration parameters."""
        return {
            'num_points': 3,  # Number of points to track
            'hsv_lower': np.array([35, 50, 50]),  # Green color lower bound
            'hsv_upper': np.array([85, 255, 255]), # Green color upper bound
            'min_contour_area': 1,
            'max_contour_area': 1000,
            'frame_average_count': 3,
            'pixel_tolerance': 5,
            'min_depth': 100,  # mm
            'output_directory': 'needle_tracking_output'
        }

    def _setup_data_storage(self):
        if not os.path.exists(self.config['output_directory']):
            os.makedirs(self.config['output_directory'])
            os.makedirs(os.path.join(self.config['output_directory'], 'images'))

        self.data_file = os.path.join(
            self.config['output_directory'],
            f'tracking_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        self.tracking_data = []

    def initialize_camera(self) -> bool:
            """Initialize the camera with correct coordinate system."""
            try:
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.HD720
                init_params.camera_fps = 30
                init_params.depth_mode = sl.DEPTH_MODE.ULTRA
                init_params.coordinate_units = sl.UNIT.MILLIMETER
                # Use standard right-handed coordinate system
                init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
                init_params.depth_minimum_distance = self.config['min_depth']

                status = self.zed.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    logger.error(f"Camera initialization failed: {status}")
                    return False

                # Set up runtime parameters
                self.runtime_params = sl.RuntimeParameters()
                self.runtime_params.confidence_threshold = 50
                self.runtime_params.texture_confidence_threshold = 100

                camera_info = self.zed.get_camera_information()
                logger.info(f"Camera baseline: {camera_info.camera_configuration.calibration_parameters.get_camera_baseline()} mm")
                logger.info(f"Focal length: {camera_info.camera_configuration.calibration_parameters.left_cam.fx} pixels")

                return True

            except Exception as e:
                logger.error(f"Error initializing camera: {str(e)}")
                return False

    def detect_needle(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], list[Optional[Tuple[int, int]]]]:
        """Detect multiple points in frame using HSV color filtering and contour detection."""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            logger.debug(f"Frame converted to HSV. Shape: {hsv.shape}")

            # Create mask using HSV thresholds
            mask = cv2.inRange(hsv, self.config['hsv_lower'], self.config['hsv_upper'])

            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter valid contours
            valid_contours = [
                cnt for cnt in contours
                if self.config['min_contour_area'] <= cv2.contourArea(cnt) <= self.config['max_contour_area']
            ]

            points = []
            # Sort contours by x-coordinate to maintain consistent labeling
            if valid_contours:
                # Get centroids for all valid contours
                centroids = []
                for contour in valid_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy, contour))

                # Sort by x-coordinate (left to right)
                centroids.sort(key=lambda x: x[0])

                # Take up to num_points contours
                for i, (cx, cy, contour) in enumerate(centroids[:self.config['num_points']]):
                    # Draw detection visualization
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # Add point label
                    cv2.putText(
                        frame,
                        f"P{i+1}",
                        (cx + 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

                    points.append((cx, cy))

            # Fill remaining points with None if we found fewer than num_points
            while len(points) < self.config['num_points']:
                points.append(None)

            return frame, points

        except Exception as e:
            logger.error(f"Error in needle detection: {str(e)}")
            return frame, [None] * self.config['num_points']

    def overlay_text(self, frame, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2):
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    def calculate_3d_coordinates(self, left_point: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Calculate 3D coordinates with Z increasing away from camera.
        """
        try:
            if not isinstance(left_point, tuple) or len(left_point) != 2:
                logger.error(f"Invalid point format: {left_point}")
                return None

            camera_info = self.zed.get_camera_information()
            baseline = camera_info.camera_configuration.calibration_parameters.get_camera_baseline()
            focal_length = camera_info.camera_configuration.calibration_parameters.left_cam.fx

            # Get right camera point using point cloud
            err = self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
            if err != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to retrieve point cloud: {err}")
                return None

            point_cloud_np = self.point_cloud.get_data()

            # Extract coordinates from point cloud using point coordinates
            x, y = left_point
            if 0 <= y < point_cloud_np.shape[0] and 0 <= x < point_cloud_np.shape[1]:
                point_data = point_cloud_np[y, x]
                X = point_data[0]  # X coordinate (right is positive)
                Y = point_data[1]  # Y coordinate (up is positive)
                Z = abs(point_data[2])  # Make Z positive away from camera

                logger.debug(f"Raw coordinates from point cloud - X: {X:.1f}, Y: {Y:.1f}, Z: {Z:.1f}")

                # Validate coordinates
                if np.isnan(Z) or np.isnan(X) or np.isnan(Y):
                    logger.debug("NaN values found in coordinates")
                    return None

                if Z < self.config['min_depth']:
                    logger.debug(f"Z distance {Z:.1f}mm is less than minimum threshold {self.config['min_depth']}mm")
                    return None

                coordinates = np.array([X, Y, Z])
                return coordinates
            else:
                logger.error(f"Point coordinates {left_point} out of bounds for point cloud shape {point_cloud_np.shape}")
                return None

        except Exception as e:
            logger.error(f"Error calculating 3D coordinates: {str(e)}")
            return None

    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a single frame from the ZED camera.
        """
        try:
            if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                logger.debug("Failed to grab new frame")
                return None, None

            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

            left_frame = self.image_left.get_data()
            right_frame = self.image_right.get_data()

            logger.debug("Processing left frame for needle detection")
            left_frame, left_points = self.detect_needle(left_frame)

            logger.debug("Processing right frame for needle detection")
            right_frame, right_points = self.detect_needle(right_frame)

            logger.debug(f"Left points detected at: {left_points}")

            # Process each detected point
            for i, left_point in enumerate(left_points):
                if left_point is not None:
                    try:
                        coords = self.calculate_3d_coordinates(left_point)
                        if coords is not None:
                            self.current_coordinates[i] = self.smooth_coordinates(coords, i)
                            logger.info(f"Point {i+1} coordinates: {self.current_coordinates[i]}")
                    except Exception as e:
                        logger.error(f"Error processing point {i+1}: {str(e)}")
                        self.current_coordinates[i] = None
                else:
                    self.current_coordinates[i] = None

            return left_frame, right_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None, None

    def save_tracking_data(self):
        try:
            # Create column names for each point
            columns = ['Timestamp']
            for i in range(self.config['num_points']):
                columns.extend([f'X{i+1}', f'Y{i+1}', f'Z{i+1}'])

            # Convert the tracking data to match the column format
            formatted_data = []
            for entry in self.tracking_data:
                timestamp = entry[0]  # First element is timestamp
                data_row = [timestamp]
                for coord in entry[1:]:  # Rest are coordinates
                    if coord is not None:
                        data_row.extend(coord)
                    else:
                        data_row.extend([np.nan, np.nan, np.nan])
                formatted_data.append(data_row)

            df = pd.DataFrame(formatted_data, columns=columns)
            df.to_excel(self.data_file, index=False)
            logger.info(f"Tracking data saved to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving tracking data: {str(e)}")
            logger.error(f"Data columns: {columns}")
            logger.error(f"Data shape: {len(self.tracking_data[0]) if self.tracking_data else 'No data'}")

    def run(self):
        if not self.initialize_camera():
            return

        self.running = True
        try:
            while self.running:
                left_frame, right_frame = self.process_frame()

                if left_frame is None or right_frame is None:
                    continue

                # Display frames
                combined_frame = np.hstack((left_frame, right_frame))

                # Add coordinate overlay for each point
                for i, coords in enumerate(self.current_coordinates):
                    if coords is not None:
                        cv2.putText(
                            combined_frame,
                            f"P{i+1}: X={coords[0]:.1f} Y={coords[1]:.1f} Z={coords[2]:.1f}",
                            (10, 30 + i*30),  # Stack coordinates vertically
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                # Add HSV adjustment instructions
                cv2.putText(
                    combined_frame,
                    "Press 'h' to adjust HSV values",
                    (10, combined_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

                cv2.imshow("Stereo Needle Tracking", combined_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    # Capture all current points if at least one is valid
                    if any(coord is not None for coord in self.current_coordinates):
                        # Create data entry with timestamp and all coordinates
                        data_entry = [datetime.now()]
                        data_entry.extend(self.current_coordinates)  # Add all coordinates

                        self.tracking_data.append(data_entry)
                        logger.info(f"Points captured: {self.current_coordinates}")

                        # Add visual feedback
                        cv2.putText(
                            combined_frame,
                            "Points Captured!",
                            (combined_frame.shape[1]//2 - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow("Stereo Needle Tracking", combined_frame)
                        cv2.waitKey(500)

                        # Save the snapshot
                        snapshot_filename = f"/home/vasanth/needle_tracking_output/images/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(snapshot_filename, cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
                        logger.info(f"Snapshot saved: {snapshot_filename}")
                    else:
                        logger.warning("No valid points to capture")
                elif key == ord('h'):
                    self._create_hsv_trackbars()

        except KeyboardInterrupt:
            logger.info("Tracking stopped by user")
        finally:
            self.cleanup()

    def _create_hsv_trackbars(self):
        """Create a visual color picker interface"""
        window_name = "Color Picker"
        cv2.namedWindow(window_name)

        class ColorPicker:
            def __init__(self, config):
                self.config = config
                self.dragging = False
                self.selected_color = None
                self.tolerance = [25, 50, 50] # HSV tolerance for color selection

            def create_color_palette(self, height=400, width=600):
                """Create HSV color palette"""
                image = np.zeros((height, width, 3), dtype=np.uint8)

                # Create HSV color space
                for x in range(width):
                    for y in range(height):
                        # Map x to Hue (0-180)
                        hue = int((x / width) * 180)
                        # Map y to Saturation and Value
                        sat = int((1 - (y / height)) * 255)
                        val = 255

                        # Set the color
                        image[y, x] = [hue, sat, val]

                # Convert to BGR for display
                return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            def update_color_info(self, hsv_color):
                """Update HSV range based on selected color"""
                h, s, v = hsv_color

                # Create range around selected color
                h_low = max(0, h - self.tolerance[0])
                h_high = min(180, h + self.tolerance[0])
                s_low = max(0, s - self.tolerance[1])
                s_high = min(255, s + self.tolerance[1])
                v_low = max(0, v - self.tolerance[2])
                v_high = min(255, v + self.tolerance[2])

                # Update config
                self.config['hsv_lower'] = np.array([h_low, s_low, v_low])
                self.config['hsv_upper'] = np.array([h_high, s_high, v_high])

                return (h_low, h_high, s_low, s_high, v_low, v_high)

        picker = ColorPicker(self.config)
        color_image = picker.create_color_palette()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get color at clicked point
                bgr_color = color_image[y, x]
                hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

                # Update HSV ranges
                h_low, h_high, s_low, s_high, v_low, v_high = picker.update_color_info(hsv_color)

                # Create info display
                info_display = np.zeros((200, 600, 3), dtype=np.uint8)

                # Show selected color
                cv2.rectangle(info_display, (10, 10), (110, 110),
                            [int(c) for c in bgr_color], -1)
                cv2.rectangle(info_display, (10, 10), (110, 110),
                            (255, 255, 255), 1)

                # Show text information
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(info_display, f"HSV: {hsv_color}",
                        (130, 40), font, 0.7, (255, 255, 255), 1)
                cv2.putText(info_display, f"Range H: {h_low}-{h_high}",
                        (130, 80), font, 0.7, (255, 255, 255), 1)
                cv2.putText(info_display, f"Range S: {s_low}-{s_high}",
                        (130, 120), font, 0.7, (255, 255, 255), 1)
                cv2.putText(info_display, f"Range V: {v_low}-{v_high}",
                        (130, 160), font, 0.7, (255, 255, 255), 1)

                # Show help text
                cv2.putText(info_display, "Click to select color. Press 'q' to close.",
                        (10, 190), font, 0.6, (200, 200, 200), 1)

                cv2.imshow("Color Info", info_display)

        # Set mouse callback
        cv2.setMouseCallback(window_name, on_mouse)

        # Main display loop
        while True:
            # Show color picker
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(window_name)
        cv2.destroyWindow("Color Info")

    def cleanup(self):
        self.save_tracking_data()
        self.zed.close()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

def main():
    # Configuration for green color detection
    config = {
        'num_points': 4,
        'hsv_lower': np.array([35, 50, 50]),    # Green color lower bound
        'hsv_upper': np.array([85, 255, 255]),  # Green color upper bound
        'min_contour_area': 1,
        'max_contour_area': 1000,
        'frame_average_count': 3,
        'pixel_tolerance': 5,
        'min_depth': 100,
        'output_directory': 'needle_tracking_output'
    }

    tracker = StereoNeedleTracker(config)
    tracker.run()

if __name__ == "__main__":
    main()