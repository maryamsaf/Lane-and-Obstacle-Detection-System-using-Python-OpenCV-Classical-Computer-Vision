import cv2
import numpy as np
import time  # <-- added for FPS/STOP timing

# --------- small helper to compute R^2 from your collected line segments (no change to detection) ---------
def _lane_r2_from_lines(lines, cols):
    if not lines or len(lines) < 2:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    if len(xs) < 3:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    try:
        deg = 2 if len(xs) >= 10 else 1  # gentle: 2nd order if we have enough points
        poly = np.polyfit(ys, xs, deg, rcond=1e-10)
        x_pred = np.polyval(poly, ys)
        ss_res = np.sum((xs - x_pred) ** 2)
        ss_tot = np.sum((xs - xs.mean()) ** 2) + 1e-9
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        return float(r2)
    except np.linalg.LinAlgError:
        return None

# Process video to detect obstacles and lanes for vehicle guidance
def thresholding_video(video_path, threshold_value=150):
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # --------- metrics (added) ---------
    total_frames = 0
    lane_center_offsets = []   # px deviation from frame center (for stability)
    left_r2_list  = []         # lane fit quality over time
    right_r2_list = []
    obstacle_frames = 0
    stop_time_s = 0.0
    last_t = time.perf_counter()
    total_walltime = 0.0

    # Loop through each video frame
    while True:
        ret, frame = cap.read()
        # End loop at video's end
        if not ret:
            break
        total_frames += 1

        # timing for FPS and STOP seconds
        now = time.perf_counter()
        dt = max(1e-6, now - last_t)
        last_t = now
        total_walltime += dt

        # Halve frame size for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rows, cols = frame.shape[:2]

        # Grayscale and preprocess for better contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Create binary image for obstacle detection
        thresh = np.where(blurred > threshold_value, 255, 0).astype(np.uint8)
        roi_start = int(rows * 0.5)
        roi_end = rows
        roi = thresh[roi_start:roi_end, :]
        white_pixels = np.sum(roi == 255)

        # Check if road is clear based on bright pixels
        min_white_pixels = 5000
        road_detected = white_pixels > min_white_pixels

        obstacle_image = frame.copy()
        obstacles_in_critical_area = False

        # Look for obstacles if road is detected
        if road_detected:
            thresh_inv = cv2.bitwise_not(thresh)
            roi_thresh_inv = thresh_inv[roi_start:roi_end, :]

            # Set obstacle detection thresholds
            min_pixel_density = 40
            height_threshold = 10
            roi_height = roi_end - roi_start

            # Define critical area for obstacles
            # (kept exactly as you had it)
            top_critical = int(roi_start + roi_height * 0.5)
            bottom_critical = int(roi_end - roi_height * 0.1)
            center_start = int(cols * 0.7)
            center_end = int(cols * 0.3)

            obstacle_regions = []

            # Scan ROI for obstacles line by line
            for y in range(roi_thresh_inv.shape[0]):
                x = 0
                while x < roi_thresh_inv.shape[1]:
                    if roi_thresh_inv[y, x] == 255:
                        start_x = x
                        while x < roi_thresh_inv.shape[1] and roi_thresh_inv[y, x] == 255:
                            x += 1
                        length = x - start_x
                        if length > min_pixel_density:
                            height = 1
                            y2 = y + 1
                            while y2 < roi_thresh_inv.shape[0]:
                                if roi_thresh_inv[y2, start_x:start_x + length].mean() > 200:
                                    height += 1
                                    y2 += 1
                                else:
                                    break
                            if height > height_threshold:
                                obstacle_regions.append([start_x, y, x, y + height])
                    else:
                        x += 1

            # Draw obstacles and check critical area
            for region in obstacle_regions:
                x1, y1, x2, y2 = region
                y1_adj = y1 + roi_start
                y2_adj = y2 + roi_start
                cv2.rectangle(obstacle_image, (x1, y1_adj), (x2, y2_adj), (0, 0, 255), 2)

                obstacle_center_x = (x1 + x2) / 2
                in_horizontal_critical = center_start <= obstacle_center_x <= center_end
                in_vertical_critical = y1_adj <= top_critical or y2_adj >= bottom_critical
                if in_horizontal_critical and in_vertical_critical:
                    obstacles_in_critical_area = True

        # Set trapezoid ROI for lane detection
        roi_vertices = np.array([[
            (cols * 0.1, rows),
            (cols * 0.4, rows * 0.6),
            (cols * 0.6, rows * 0.6),
            (cols * 0.9, rows)
        ]], dtype=np.int32)

        # Mask grayscale image for lane ROI
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_image = cv2.bitwise_and(gray, mask)

        # Detect edges in masked region
        edges = cv2.Canny(masked_image, 50, 150)

        # Find lines with Hough Transform
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40,
                                minLineLength=30, maxLineGap=100)

        lane_image = frame.copy()
        left_lines = []
        right_lines = []

        # Sort lines into left/right lanes by slope
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if -0.9 < slope < -0.2:
                        left_lines.append([x1, y1, x2, y2])
                    elif 0.2 < slope < 0.9:
                        right_lines.append([x1, y1, x2, y2])

        # --- R^2 from your raw line points (doesn't affect detection)
        left_r2  = _lane_r2_from_lines(left_lines,  cols)
        right_r2 = _lane_r2_from_lines(right_lines, cols)
        if left_r2  is not None: left_r2_list.append(left_r2)
        if right_r2 is not None: right_r2_list.append(right_r2)

        # Fit smooth line to lane segments
        def fit_lane_lines(lines, y_range):
            if not lines or len(lines) < 2:
                return None

            x_coords = []
            y_coords = []

            for x1, y1, x2, y2 in lines:
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

            if len(x_coords) < 3:
                return None

            try:
                if len(x_coords) < 5:
                    poly = np.polyfit(y_coords, x_coords, 1)
                    y_vals = np.linspace(y_range[0], y_range[1], 100)
                    x_vals = poly[0] * y_vals + poly[1]
                else:
                    poly = np.polyfit(y_coords, x_coords, 2, rcond=1e-10)
                    y_vals = np.linspace(y_range[0], y_range[1], 100)
                    x_vals = poly[0] * y_vals ** 2 + poly[1] * y_vals + poly[2]

                valid_points = (x_vals >= 0) & (x_vals < cols)
                if not np.any(valid_points):
                    return None

                return np.column_stack((x_vals[valid_points], y_vals[valid_points])).astype(np.int32)
            except np.linalg.LinAlgError:
                try:
                    poly = np.polyfit(y_coords, x_coords, 1)
                    y_vals = np.linspace(y_range[0], y_range[1], 100)
                    x_vals = poly[0] * y_vals + poly[1]

                    valid_points = (x_vals >= 0) & (x_vals < cols)
                    if not np.any(valid_points):
                        return None

                    return np.column_stack((x_vals[valid_points], y_vals[valid_points])).astype(np.int32)
                except:
                    return None

        # Fit lanes in bottom part of frame
        y_range = (rows * 0.6, rows)
        left_lane = fit_lane_lines(left_lines, y_range)
        right_lane = fit_lane_lines(right_lines, y_range)

        # Draw detected lanes
        if left_lane is not None:
            cv2.polylines(lane_image, [left_lane], isClosed=False, color=(255, 0, 0), thickness=2)
        if right_lane is not None:
            cv2.polylines(lane_image, [right_lane], isClosed=False, color=(255, 0, 0), thickness=2)

        # Calculate steering direction
        direction = "Forward"
        if left_lane is not None and right_lane is not None:
            try:
                if len(left_lane) > 0 and len(right_lane) > 0:
                    bottom_left_x = np.mean(left_lane[-5:, 0]) if len(left_lane) >= 5 else left_lane[-1, 0]
                    bottom_right_x = np.mean(right_lane[-5:, 0]) if len(right_lane) >= 5 else right_lane[-1, 0]

                    lane_center = (bottom_left_x + bottom_right_x) / 2
                    frame_center = cols / 2
                    deviation = lane_center - frame_center

                    # ---- add to stability metric (only when both lanes exist)
                    lane_center_offsets.append(float(deviation))

                    cv2.circle(lane_image, (int(lane_center), rows - 10), 5, (0, 255, 255), -1)
                    cv2.circle(lane_image, (int(frame_center), rows - 10), 5, (255, 255, 0), -1)

                    threshold = cols * 0.05
                    if deviation < -threshold:
                        direction = "Right"
                    elif deviation > threshold:
                        direction = "Left"
            except Exception as e:
                print(f"Direction calculation error: {e}")
                direction = "Forward"
        elif left_lane is not None:
            direction = "Right"
        elif right_lane is not None:
            direction = "Left"

        # Draw ROI and critical areas
        roi_image = frame.copy()
        cv2.rectangle(roi_image, (0, roi_start), (cols, roi_end), (0, 255, 0), 2)
        cv2.line(roi_image, (center_start, top_critical), (center_end, top_critical), (255, 0, 0), 2)
        cv2.line(roi_image, (center_start, bottom_critical), (center_end, bottom_critical), (255, 0, 0), 2)
        cv2.line(roi_image, (center_start, top_critical), (center_start, bottom_critical), (255, 0, 0), 2)
        cv2.line(roi_image, (center_end, top_critical), (center_end, bottom_critical), (255, 0, 0), 2)

        # Show stop/move and steering direction
        status_text = "STOP" if obstacles_in_critical_area else "MOVE"
        cv2.putText(roi_image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255) if obstacles_in_critical_area else (0, 255, 0), 3)
        cv2.putText(roi_image, f"Direction: {direction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0), 2)

        # Display processed images
        cv2.imshow('Edges', edges)
        cv2.imshow('Lane Detection', lane_image)
        cv2.imshow('ROI & Critical Areas', roi_image)

        # ---- metrics update for obstacles / STOP
        if obstacles_in_critical_area:
            obstacle_frames += 1
            stop_time_s += dt

        # Exit on ESC key
        key = cv2.waitKey(1)  # (kept responsive)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ------------------- SUMMARY YOU CAN PUT ON CV -------------------
    avg_fps = (total_frames / total_walltime) if total_walltime > 0 else None
    lane_stability = float(np.std(lane_center_offsets)) if len(lane_center_offsets) >= 2 else None
    left_r2_mean  = float(np.mean(left_r2_list))  if left_r2_list  else None
    right_r2_mean = float(np.mean(right_r2_list)) if right_r2_list else None
    obstacle_persistence_pct = (obstacle_frames / max(1, total_frames)) * 100.0

    print("\n=== RUN SUMMARY (copy to CV/README) ===")
    print(f"Frames processed:             {total_frames}")
    if avg_fps is not None:
        print(f"Average FPS (measured):       {avg_fps:.1f}")
    else:
        print(f"Average FPS (measured):       N/A")
    if lane_stability is not None:
        print(f"Lane stability (px std-dev):  {lane_stability:.2f}  (lower = steadier)")
    else:
        print(f"Lane stability (px std-dev):  N/A (both lanes rarely visible)")
    if left_r2_mean is not None or right_r2_mean is not None:
        print(f"Lane fit R^2 (L/R):           "
              f"{left_r2_mean if left_r2_mean is not None else float('nan'):.2f} / "
              f"{right_r2_mean if right_r2_mean is not None else float('nan'):.2f}  (0–1)")
    else:
        print(f"Lane fit R^2 (L/R):           N/A")
    print(f"Obstacle persistence:         {obstacle_persistence_pct:.2f}% of frames")
    print(f"STOP time:                    {stop_time_s:.2f} s")


if __name__ == "__main__":
    video_path = r'DIP Project Videos/outside.mp4'  # use r'' or double backslashes on Windows
    thresholding_video(video_path, threshold_value=150)
