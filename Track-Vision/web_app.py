from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from utils.face_encoder import FaceEncoder
from utils.camera_handler import CameraHandler
from utils.face_matcher import FaceMatcher
import os
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_photos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Global variables
camera = None
matcher = None
reference_encoding = None
camera_lock = threading.Lock()
is_processing = False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
    """
    Generator function to yield video frames for streaming.
    """
    global camera, matcher, is_processing
    
    while True:
        if camera is None or not camera.is_running():
            # Send a placeholder frame if camera is not ready
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera Not Ready", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
        
        with camera_lock:
            success, frame = camera.get_frame()
        
        if not success:
            continue
        
        # Process frame if matcher is available
        if matcher is not None and is_processing:
            annotated_frame, match_results = matcher.process_frame(frame)
            
            # Add status text
            if any(result['is_match'] for result in match_results):
                status = "TARGET FOUND!"
                color = (0, 255, 0)
            elif len(match_results) > 0:
                status = "Faces Detected (No Match)"
                color = (0, 165, 255)
            else:
                status = "Searching..."
                color = (255, 255, 255)
            
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Upload reference photo to start", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index_new.html')

@app.route('/upload', methods=['POST'])
def upload_reference():
    """Handle reference photo upload."""
    global reference_encoding, matcher, is_processing
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image and extract encoding
        encoder = FaceEncoder(model="hog")
        image, face_locations, face_encoding = encoder.process_uploaded_photo(filepath)
        
        if face_encoding is None:
            return jsonify({
                'success': False, 
                'message': 'No face detected in the uploaded photo'
            }), 400
        
        # Store encoding
        reference_encoding = face_encoding
        
        # Get tolerance from request or use default
        tolerance = float(request.form.get('tolerance', 0.6))
        
        # Initialize matcher
        matcher = FaceMatcher(
            reference_encoding=reference_encoding,
            tolerance=tolerance,
            detection_model="hog",
            enable_parallel=True
        )
        
        is_processing = True
        
        return jsonify({
            'success': True,
            'message': 'Reference photo processed successfully',
            'faces_detected': len(face_locations)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera feed."""
    global camera
    
    data = request.get_json()
    camera_type = data.get('camera_type', 'webcam')
    ip_url = data.get('ip_url', None)
    
    try:
        with camera_lock:
            # Stop existing camera if any
            if camera is not None:
                camera.release()
            
            # Initialize new camera
            camera = CameraHandler(camera_type=camera_type, ip_url=ip_url)
            
            if not camera.initialize():
                return jsonify({
                    'success': False,
                    'message': 'Failed to initialize camera'
                }), 500
        
        return jsonify({
            'success': True,
            'message': 'Camera started successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting camera: {str(e)}'
        }), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera feed."""
    global camera, is_processing
    
    try:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        is_processing = False
        
        return jsonify({
            'success': True,
            'message': 'Camera stopped successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping camera: {str(e)}'
        }), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing', methods=['POST'])
def toggle_processing():
    """Toggle face recognition processing."""
    global is_processing
    
    is_processing = not is_processing
    
    return jsonify({
        'success': True,
        'is_processing': is_processing
    })

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get matching statistics."""
    global matcher
    
    if matcher is None:
        return jsonify({
            'success': False,
            'message': 'Matcher not initialized'
        }), 400
    
    stats = matcher.get_statistics()
    
    return jsonify({
        'success': True,
        'stats': stats
    })

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset matching statistics."""
    global matcher
    
    if matcher is not None:
        matcher.reset_statistics()
    
    return jsonify({
        'success': True,
        'message': 'Statistics reset'
    })

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8800, threaded=True)
