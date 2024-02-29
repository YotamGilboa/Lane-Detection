import cv2
import numpy as np
from collections import deque
import time


# Initialize variables for simple moving average
alpha = 0.2  # Smoothing factor
prev_left_line = None
prev_right_line = None

# Load indicator images
left_indicator = cv2.imread('Left.png')
right_indicator = cv2.imread('Right.png')

# Resize indicator images to 100x100 px
left_indicator = cv2.resize(left_indicator, (100, 100))
right_indicator = cv2.resize(right_indicator, (100, 100))

indicator_duration = 2  # Duration to show the indicator in seconds
indicator_start_time_r = 0
indicator_start_time_l = 0

def process_frame(frame):
    global prev_left_line, prev_right_line, indicator_start_time

    # Resize the frame
    resized_frame = cv2.resize(frame, (1000, 700))

    # Convert to HSV for color filtering
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for yellow and white
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([255, 30, 255], dtype=np.uint8)

    # Create masks for yellow and white
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)

    # Combine masks
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Combine color and grayscale masks
    combined_mask = cv2.bitwise_or(color_mask, blurred_frame)

    # Define region of interest (ROI)
    roi_bottom_left = (100, 630)
    roi_top_left = (450, 400)
    roi_top_right = (600, 400)
    roi_bottom_right = (1100, 550)
    
    roi_vertices = np.array([roi_bottom_left, roi_top_left, roi_top_right, roi_bottom_right], np.int32)
    mask = np.zeros_like(combined_mask)
    cv2.fillPoly(mask, [roi_vertices], 255)
    roi_frame = cv2.bitwise_and(combined_mask, mask)

    # Apply Canny edge detector
    edges = cv2.Canny(roi_frame, 100, 150)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=35, maxLineGap=55)

    # Extrapolate lines and draw on the black background
    black_background = np.zeros_like(resized_frame)

    # Check if lines exist and calculate positions
    if lines is not None:
        left_line, right_line = extrapolate_lines(lines, 630, 400)

        # Initialize direction as 'Straight'
        direction = 'Straight'
        
        max_right_decision_point = resized_frame.shape[0] * 0.8     #Dinamic
        min_right_decision_point = resized_frame.shape[0] * 0.58
        
        #moving left
        if left_line is not None and len(left_line) > 0 and direction != 'Right':
            left_line_bottom_x = left_line[0][0]
            #print("Left: ", left_line_bottom_x)
            if (left_line_bottom_x > 320):
                if right_line is None or (right_line is not None and len(right_line) > 0 and right_line[0][0] > 980):
                    direction = 'Left'
                    #print("left beaches:", left_line_bottom_x)

        #moving right
        if right_line is not None and len(right_line) > 0 and direction != 'Left':
            right_line_bottom_x = right_line[0][0]
            #print("right:", right_line_bottom_x)
            if (right_line_bottom_x < 750):
                #print("", "less than 700")
                if left_line is None or (left_line is not None and len(left_line) > 0 and left_line[0][0] < 450):
                    #print("", "less than 450")
                    direction = 'Right'
        
        # Display direction indicators
        if direction == 'Left':
            resized_frame[0:100, 0:100] = left_indicator

        # Display direction indicators
        if direction == 'Left':
            resized_frame[0:100, 0:100] = left_indicator
            indicator_start_time_l = time.time()  # Update the start time
            #print("start time Left: ", indicator_start_time)
            
        if direction == 'Right':
            resized_frame[0:100, resized_frame.shape[1]-100:resized_frame.shape[1]] = right_indicator
            indicator_start_time_r = time.time()  # Update the start time
            print("start time Right: ", indicator_start_time_r)
        
        # Smooth lines
        left_line = smooth_line(left_line, prev_left_line, alpha)
        right_line = smooth_line(right_line, prev_right_line, alpha)

        # Draw lines
        if left_line is not None:
            draw_extrapolated_line(black_background, left_line, (0, 0, 255), 6)
        if right_line is not None:
            draw_extrapolated_line(black_background, right_line, (0, 255, 0), 6)

        # Update previous lines
        prev_left_line = left_line
        prev_right_line = right_line
        
        #if dominant_direction is not None:
            #direction_history.append(dominant_direction)

        # Combine the black background with the original frame
        result_frame = cv2.addWeighted(resized_frame, 1, black_background, 1, 0)
        return result_frame

def extrapolate_lines(lines, y_bottom, y_top):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check for vertical lines
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if slope > 0.5:  # Right lane line
            right_lines.append(line)
        elif slope < -0.7:  # Adjust this threshold for the left lane line
            left_lines.append(line)

    left_line = average_slope_intercept(left_lines, y_bottom, y_top)
    right_line = average_slope_intercept(right_lines, y_bottom, y_top)

    return left_line, right_line

def average_slope_intercept(lines, y_bottom, y_top):
    if len(lines) > 0:
        slopes = []
        intercepts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check for vertical lines
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            slopes.append(slope)
            intercepts.append(intercept)

        avg_slope = np.nanmean(slopes)  # Use np.nanmean to handle NaN values
        avg_intercept = np.nanmean(intercepts)

        x_bottom = int((y_bottom - avg_intercept) / avg_slope)
        x_top = int((y_top - avg_intercept) / avg_slope)

        return [(x_bottom, y_bottom, x_top, y_top)]

    return []

def draw_extrapolated_line(image, line, color, thickness):
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def smooth_line(current_line, prev_line, alpha):
    if current_line and prev_line:
        smoothed_line = []
        for cur_point, prev_point in zip(current_line[0], prev_line[0]):
            smoothed_point = int(alpha * cur_point + (1 - alpha) * prev_point)
            smoothed_line.append(smoothed_point)
        return [tuple(smoothed_line)]

    return current_line

# Create a VideoCapture object and read from the input file
cap = cv2.VideoCapture('videoName.mp4')    #Replace the videoName with the name of your video from the same repository

#Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1000,700))

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read until the video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        result_frame = process_frame(frame)

        out.write(result_frame)
        # Display the resulting frame
        cv2.imshow('Frame', result_frame)

        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object
cap.release()

out.release()

# Close all the frames
cv2.destroyAllWindows()
