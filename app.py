from flask import Flask, render_template, request, jsonify, Response, flash
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from flask_socketio import emit
from ultralytics import YOLO
app = Flask(__name__)
socketio = SocketIO(app)

# Absolute path to the directory containing videos
VIDEO_DIRECTORY = r'D:\AIML\smoke\smoke_latest\smoke_web\flask_folder\static\videos'

# Absolute path to the directory to save frames
SAVED_FRAMES_DIRECTORY = r'D:\AIML\smoke\smoke_latest\smoke_web\flask_folder\static\saved_frames'

# Absolute path to the directory to save processed videos
PROCESSED_VIDEOS_DIRECTORY = r'D:\AIML\smoke\smoke_latest\smoke_web\flask_folder\static\processed_videos'

# Initialize global variable for path lines
path_lines = []

# Assuming yolo_model is initialized elsewhere
yolo_model = YOLO(r'D:\AIML\smoke\smoke_latest\smoke_web\flask_folder\best3.pt')

def get_fifth_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None
    frame_number = 0
    while frame_number < 5:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            return None
        frame_number += 1
    cap.release()
    return frame

def detect_smoke(frame):
    results = yolo_model(frame)
    detected_boxes = []

    for result in results:
        for box in result.boxes:
            if box.conf > 0.35 and box.cls == 0:  # Assuming class_id 0 is smoke
                x, y, w, h = box.xywh[0]
                detected_boxes.append((int(x - w/2), int(y - h/2), int(w), int(h)))
                cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), (int(y + h/2))), (0, 255, 0), 2)
    
    # Print detected smoke coordinates
    print("Detected smoke coordinates:", detected_boxes)
    return detected_boxes

def draw_path_lines(frame, path_lines):
    for line in path_lines:
        for i in range(len(line) - 1):
            pt1 = (int(line[i][0]), int(line[i][1]))
            pt2 = (int(line[i + 1][0]), int(line[i + 1][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.circle(frame, pt1, 5, (255, 0, 0), -1)
        pt_last = (int(line[-1][0]), int(line[-1][1]))
        cv2.circle(frame, pt_last, 5, (255, 0, 0), -1)

def increase_contrast(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrast_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return contrast_frame

def check_smoke_flow_within_path(smoke_coordinates, path_lines):
    # Ensure path_lines form a closed contour
    contour_points = np.vstack([path_lines[0], path_lines[1][::-1], [path_lines[0][0]]]).astype(np.float32)
    contour = contour_points.reshape((-1, 1, 2))

    # Print contour points
    print("Contour points:", contour_points)

    for coord in smoke_coordinates:
        # Correctly order the coordinates as (x, y) for pointPolygonTest
        coord_tuple = (float(coord[0]), float(coord[1]))
        result = cv2.pointPolygonTest(contour, coord_tuple, False)
        
        # Print each coordinate and the result of point-in-polygon test
        print(f"Checking point {coord_tuple} result: {result}")
        if result < 0:
            print(f"Point {coord_tuple} is outside the path contour.")
            return False
        else:
            return True
    

def process_video_stream(input_path, path_lines, deviation_detected, blur_ksize=(21, 21), blur_sigma=0, threshold=15):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    deviation_detected = False
    frame_width = 800
    frame_height = 500
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_frame_data = None
    deviation_timer = 0
    deviation_threshold = 4 * fps
    deviations_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        # Detect smoke
        detected_boxes = detect_smoke(frame)

        for box in detected_boxes:
            x, y, w, h = box
            roi = frame[y:y + h, x:x + w]

            # Apply contrast increase, gray conversion, and Gaussian blur
            contrast_frame = increase_contrast(roi)
            gray_frame = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, blur_ksize, blur_sigma)

            if prev_frame_data is None:
                delta_frame = np.zeros_like(blurred_frame)
                thresholded_frame = np.zeros_like(blurred_frame)
            else:
                prev_frame_gray = cv2.cvtColor(prev_frame_data, cv2.COLOR_BGR2GRAY)
                prev_roi = prev_frame_gray[y:y + h, x:x + w]
                delta_frame = cv2.absdiff(blurred_frame, prev_roi)
                _, thresholded_frame = cv2.threshold(delta_frame, threshold, 255, cv2.THRESH_BINARY)
                white_pixels = np.column_stack(np.where(thresholded_frame == 255))

                # Adjust white_pixels coordinates to be relative to the frame
                adjusted_white_pixels = [(pixel[1] + x, pixel[0] + y) for pixel in white_pixels]

                # Include the bounding box coordinates in the adjusted_white_pixels
                if check_smoke_flow_within_path(adjusted_white_pixels, path_lines):
                    deviation_timer = 0
                else:
                    deviation_timer += 1

                if deviation_timer >= deviation_threshold:
                    print("Smoke deviation detected!")
                    app.logger.debug('Test event received with data: %s', 'Smoke Deviation Detected!')
                    emit('deviation_detected', {'data': 'Smoke deviation detected!'})
                    app.logger.debug('Deviation detected event emitted')
                    deviations_found = True
                    deviation_detected = True
                    socketio.emit('deviation_detected', {'data': 'Smoke deviation detected!'})
                    deviation_timer = 0

                    # Print detection message in the backend
                    print("Deviation detected in the backend.")

                for pixel in adjusted_white_pixels:
                    cv2.circle(frame, (pixel[0], pixel[1]), 3, (0, 255, 0), -1)

            frame[y:y + h, x:x + w] = roi

        draw_path_lines(frame, path_lines)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        prev_frame_data = frame.copy()  # Store the frame data for the next iteration

    cap.release()
    cv2.destroyAllWindows()

    if not deviations_found:
        print("Smoke within range")

    return deviation_detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_name = request.args.get('video_name')
    if not video_name:
        return "Video name not provided.", 400

    video_path = os.path.join(VIDEO_DIRECTORY, video_name)
    if not os.path.exists(video_path):
        return f"Video {video_name} not found.", 404

    return Response(process_video_stream(video_path, path_lines, deviation_detected=False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    global path_lines
    data = request.json
    first_line = data.get('first_line', [])
    second_line = data.get('second_line', [])

    path_lines = [list(map(tuple, first_line)), list(map(tuple, second_line))]
    print("Path Lines:", path_lines)
    
    return jsonify({"status": "success", "message": "Coordinates saved."})

@app.route('/process_video', methods=['POST'])
def process_video_route():
    global path_lines
    data = request.json
    video_name = data.get('video')
    video_path = os.path.join(VIDEO_DIRECTORY, video_name)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return jsonify({"status": "error", "message": f"Video file {video_name} does not exist."}), 404

    result = process_video_stream(video_path, path_lines, deviation_detected=False)
    deviation_detected = result

    if deviation_detected:
        app.logger.debug('Test event received with data: %s', data)
        emit('deviation_detected', {'data': data})
        app.logger.debug('Deviation detected event emitted')
        print("Smoke deviation detected in the backend!")  # Print backend message

    return jsonify({"deviation_detected": deviation_detected})

@app.route('/select_video', methods=['POST'])
def select_video():
    if 'video_file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    video_file = request.files['video_file']

    if video_file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join(VIDEO_DIRECTORY, video_filename)
    video_file.save(video_path)
    
    frame = get_fifth_frame(video_path)
    if frame is None:
        print("Error: Could not capture the 5th frame.")
        return "Error: Could not capture the 5th frame.", 400

    frame_filename = f"{video_filename.split('.')[0]}_fifth_frame.jpg"
    frame_path = os.path.join(SAVED_FRAMES_DIRECTORY, frame_filename)
    cv2.imwrite(frame_path, frame)

    return render_template('define_path.html', frame_filename=frame_filename)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8000, use_reloader=False, **{'worker_class': 'eventlet'})



