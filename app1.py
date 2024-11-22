from flask import Flask, render_template, request, jsonify, Response, url_for, flash
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from flask_socketio import emit

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
    contour_points = np.vstack([path_lines[0], path_lines[1][::-1]]).astype(np.float32)
    contour = contour_points.reshape((-1, 1, 2))

    for coord in smoke_coordinates:
        coord_tuple = (float(coord[1]), float(coord[0]))
        if cv2.pointPolygonTest(contour, coord_tuple, False) < 0:
            return False
    return True

def process_video_stream(input_path, path_lines,deviation_detected, blur_ksize=(21, 21), blur_sigma=0, threshold=5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return False
    
    deviation_detected = False

    frame_width = 800
    frame_height = 500
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_frame = None
    deviation_timer = 0
    deviation_threshold = 4 * fps
    deviations_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        contrast_frame = increase_contrast(frame)
        gray_frame = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY)
        

        if not (isinstance(blur_ksize, tuple) and len(blur_ksize) == 2 and all(isinstance(x, int) and x % 2 == 1 for x in blur_ksize)):
            blur_ksize = (21, 21)

        blurred_frame = cv2.GaussianBlur(gray_frame, blur_ksize, blur_sigma)

        if prev_frame is None:
            delta_frame = np.zeros_like(blurred_frame)
            thresholded_frame = np.zeros_like(blurred_frame)
            
        else:
            delta_frame = cv2.absdiff(blurred_frame, prev_frame)
            _, thresholded_frame = cv2.threshold(delta_frame, threshold, 255, cv2.THRESH_BINARY)
            white_pixels = np.column_stack(np.where(thresholded_frame == 255))
            
            if check_smoke_flow_within_path(white_pixels, path_lines):
                deviation_timer = 0
            else:
                deviation_timer += 1

            if deviation_timer >= deviation_threshold:
                print("Smoke deviation detected!")  # Add debug statement
                deviations_found = True
                deviation_detected = True 
                socketio.emit('deviation_detected', {'data': 'Smoke deviation detected!'})# Set deviation_detected to True
                deviation_timer = 0
                # Trigger an alert here

            for pixel in white_pixels:
                cv2.circle(frame, (pixel[1], pixel[0]), 3, (0, 255, 0), -1)

        draw_path_lines(frame, path_lines)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        prev_frame = blurred_frame
        
        
    cap.release()
    cv2.destroyAllWindows()
    
    if not deviations_found:
        print("Smoke within range")
    
    print(deviation_detected)
    
    return deviation_detected  # Return deviation_detected flag



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

    return Response(process_video_stream(video_path, path_lines,deviation_detected=False), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    global path_lines
    data = request.json
    first_line = data.get('first_line', [])
    second_line = data.get('second_line', [])

    path_lines = [list(map(tuple, first_line)), list(map(tuple, second_line))]
    print("Path Lines:", path_lines)
    
    return jsonify({"status": "success", "message": "Coordinates saved."})

from flask import Flask, render_template, request, jsonify, Response, url_for, flash

@app.route('/process_video', methods=['POST'])
def process_video_route():
    global path_lines
    data = request.json
    video_name = data.get('video')
    video_path = os.path.join(VIDEO_DIRECTORY, video_name)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return jsonify({"status": "error", "message": f"Video file {video_name} does not exist."}), 404

    result = process_video_stream(video_path, path_lines)
    deviation_detected = result.get('deviation_detected', False)

    if deviation_detected:
        socketio.emit('deviation_detected', {'message': 'Smoke deviation detected!'})

    return jsonify(result)


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
    app.run(debug=True)
