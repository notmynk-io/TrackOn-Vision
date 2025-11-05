from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import threading
import time
import sounddevice as sd

from audio_analyzer import AudioAnalyzer, sanitize_numeric

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables
camera = None
session_data = {
    'active': False,
    'subject_id': None,
    'session_id': None,
    'baseline_mode': False,
    'baseline_captured': False,
    'baseline_frame_count': 0,
    'analyzing': False,
    'emotion_log': [],
    'baseline_emotions': [],
    'current_frame': None,
    'current_emotion': 'none',
    'current_emotion_confidence': 0.0,
    'emotion_stats': {},
    'pose_insights': {},
    'eye_insights': {},
    'gesture_insights': {},
    'audio_available': False,
    'audio_current': {},
    'audio_log': [],
    'audio_baseline': [],
    'audio_baseline_stats': {},
    'start_time': None
}

# Load emotion detection model
emotion_model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Face cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


class BaseAnalyzer:
    name = "base"

    def process(self, frame, timestamp, session_state):
        raise NotImplementedError


class EmotionAnalyzer(BaseAnalyzer):
    name = "emotion"

    def process(self, frame, timestamp, session_state):
        result = detect_emotion(frame)
        if not result:
            return None

        session_state['current_emotion'] = result['emotion']
        session_state['current_emotion_confidence'] = result['confidence']
        session_state['emotion_stats'] = result['all_predictions']

        if session_state['baseline_mode']:
            session_state['baseline_emotions'].append(result['emotion'])
            session_state['baseline_frame_count'] += 1

        return result


class PoseAnalyzer(BaseAnalyzer):
    name = "pose"

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame, timestamp, session_state):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        insights = {'nose_y': None}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            left_shoulder = np.array([
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
            ])
            right_shoulder = np.array([
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
            ])
            hip = np.array([
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
            ])
            nose = np.array([
                landmarks[mp_pose.PoseLandmark.NOSE.value].x * w,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y * h
            ])

            shoulder_diff = float(abs(left_shoulder[1] - right_shoulder[1]))
            torso_vector = hip - ((left_shoulder + right_shoulder) / 2)
            vertical = np.array([0, 1])
            lean_angle = calculate_angle(hip, hip + vertical, hip + torso_vector)

            head_tilt = calculate_angle(left_shoulder, nose, right_shoulder)

            insights.update({
                'shoulder_asymmetry': shoulder_diff,
                'lean_forward': bool(lean_angle > 100),
                'lean_backward': bool(lean_angle < 80),
                'head_tilt_deg': float(head_tilt - 90),
                'nose_y': float(nose[1])
            })

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        session_state.setdefault('pose_insights', {}).update(insights)
        return insights


class EyeBehaviorAnalyzer(BaseAnalyzer):
    name = "eye_behavior"

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_blink_time = None
        self.blink_count = 0

    def reset(self):
        self.last_blink_time = None
        self.blink_count = 0

    def process(self, frame, timestamp, session_state):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        insights = {}

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]

            def eye_aspect_ratio(indices):
                points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in indices])
                vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
                horizontal = np.linalg.norm(points[0] - points[3])
                if horizontal == 0:
                    return 0
                return vertical / (2.0 * horizontal)

            left_ear = eye_aspect_ratio(left_eye_idx)
            right_ear = eye_aspect_ratio(right_eye_idx)
            ear = (left_ear + right_ear) / 2

            blink_threshold = 0.18
            is_blink = ear < blink_threshold

            current_time = datetime.now().timestamp()
            if is_blink:
                if not self.last_blink_time or current_time - self.last_blink_time > 0.2:
                    self.blink_count += 1
                    self.last_blink_time = current_time

            insights = {
                'blink_count': self.blink_count,
                'eye_aspect_ratio': ear
            }

        session_state.setdefault('eye_insights', {}).update(insights)
        return insights


class GestureAnalyzer(BaseAnalyzer):
    name = "gesture"

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def process(self, frame, timestamp, session_state):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        insights = {
            'hand_near_face': False,
            'hand_count': 0
        }

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            insights['hand_count'] = int(len(results.multi_hand_landmarks))

            nose_y = session_state.get('pose_insights', {}).get('nose_y')

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if nose_y is not None:
                    wrist_y = hand_landmarks.landmark[0].y * h
                    if wrist_y < nose_y + 40:
                        insights['hand_near_face'] = True

        session_state.setdefault('gesture_insights', {}).update(insights)
        return insights


analyzers = [
    EmotionAnalyzer(),
    PoseAnalyzer(),
    EyeBehaviorAnalyzer(),
    GestureAnalyzer()
]

audio_analyzer = AudioAnalyzer()


def handle_audio_features(features):
    """Callback invoked by AudioAnalyzer with freshly computed features."""
    feature_dict = features.as_dict()
    sanitized = {k: sanitize_numeric(v) if k != 'timestamp' else v for k, v in feature_dict.items()}

    session_data['audio_current'] = {k: v for k, v in sanitized.items() if k != 'timestamp'}
    session_data['audio_current']['timestamp'] = sanitized['timestamp']
    session_data['audio_available'] = True

    if session_data.get('baseline_mode'):
        session_data['audio_baseline'].append({k: v for k, v in session_data['audio_current'].items() if k != 'timestamp'})

    if session_data.get('analyzing'):
        session_data['audio_log'].append(sanitized)


audio_analyzer._emit = handle_audio_features  # type: ignore[attr-defined]


def compute_audio_baseline(entries):
    """Aggregate baseline audio samples into summary statistics."""
    if not entries:
        return {}

    keys = entries[0].keys()
    summary = {}
    for key in keys:
        values = [entry.get(key, 0.0) for entry in entries if isinstance(entry.get(key), (int, float))]
        if not values:
            continue
        summary[f'{key}_avg'] = round(float(np.mean(values)), 3)
        summary[f'{key}_std'] = round(float(np.std(values)), 3)
    summary['sample_count'] = len(entries)
    return summary

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def detect_emotion(frame):
    """Detect emotion from frame"""
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = faces[0]
        face_roi = gray_frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face.astype("float32") / 255.0

        # DeepFace emotion model expects 3 channels; tile grayscale frame
        reshaped_face = np.repeat(normalized_face[..., np.newaxis], 3, axis=-1)
        reshaped_face = reshaped_face.reshape(1, 48, 48, 3)
        
        raw_preds = emotion_model.predict(reshaped_face)
        preds = np.asarray(raw_preds).flatten()

        if preds.size != len(emotion_labels):
            print(f"Emotion detection warning: unexpected prediction shape {np.asarray(raw_preds).shape}")
            return None

        emotion_idx = int(np.argmax(preds))
        emotion = emotion_labels[emotion_idx]
        confidence = float(preds[emotion_idx])
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': {emotion_labels[i]: float(preds[i]) for i in range(len(preds))},
            'bbox': (int(x), int(y), int(w), int(h))
        }
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return None

def generate_frames():
    """Generate video frames with modular analyzers"""
    global camera, session_data
    
    camera = cv2.VideoCapture(0)
    frame_index = 0
    session_data.setdefault('analysis_frames', [])
    session_data['analysis_frames'].clear()

    while camera and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        frame_index += 1
        timestamp = datetime.now().isoformat()

        overlay_frame = frame.copy()
        analyzer_results = {}

        for analyzer in analyzers:
            try:
                result = analyzer.process(overlay_frame, timestamp, session_data)
                analyzer_results[analyzer.name] = result
            except Exception as exc:
                print(f"Analyzer {analyzer.name} error: {exc}")

        emotion_result = analyzer_results.get('emotion')
        if emotion_result and 'bbox' in emotion_result:
            x, y, w, h = emotion_result['bbox']
            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (66, 124, 255), 2)
            cv2.putText(
                overlay_frame,
                f"{emotion_result['emotion']}: {emotion_result['confidence']:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (66, 124, 255),
                2
            )

        if session_data['analyzing'] and frame_index % 5 == 0 and emotion_result:
            log_entry = {
                'timestamp': timestamp,
                'frame': frame_index,
                'emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                **emotion_result['all_predictions'],
            }
            log_entry.update(session_data.get('pose_insights', {}))
            log_entry.update(session_data.get('eye_insights', {}))
            log_entry.update(session_data.get('gesture_insights', {}))
            session_data['emotion_log'].append(log_entry)

        ret, buffer = cv2.imencode('.jpg', overlay_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if camera:
        camera.release()

@app.route('/')
def index():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new analysis session"""
    data = request.json
    subject_id = data.get('subject_id', 'UNKNOWN')
    
    session_data['active'] = True
    session_data['subject_id'] = subject_id
    session_data['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data['emotion_log'] = []
    session_data['baseline_emotions'] = []
    session_data['baseline_frame_count'] = 0
    session_data['baseline_captured'] = False
    session_data['emotion_stats'] = {}
    session_data['pose_insights'] = {}
    session_data['eye_insights'] = {}
    session_data['gesture_insights'] = {}
    session_data['audio_current'] = {}
    session_data['audio_log'] = []
    session_data['audio_baseline'] = []
    session_data['audio_baseline_stats'] = {}
    session_data['current_emotion'] = 'none'
    session_data['current_emotion_confidence'] = 0.0
    session_data['start_time'] = datetime.now()

    for analyzer in analyzers:
        if hasattr(analyzer, 'reset'):
            analyzer.reset()

    audio_analyzer.reset()
    session_data['audio_available'] = audio_analyzer.start()

    return jsonify({
        'success': True,
        'session_id': session_data['session_id'],
        'message': 'Session started'
    })

@app.route('/start_baseline', methods=['POST'])
def start_baseline():
    """Start baseline capture"""
    if not session_data['active']:
        return jsonify({'success': False, 'error': 'No active session'})
    
    session_data['baseline_mode'] = True
    session_data['baseline_captured'] = False
    session_data['baseline_frame_count'] = 0
    session_data['baseline_emotions'] = []
    session_data['audio_baseline'] = []
    session_data['audio_baseline_stats'] = {}
    
    for analyzer in analyzers:
        if hasattr(analyzer, 'reset'):
            analyzer.reset()

    audio_analyzer.reset()
    session_data['audio_available'] = audio_analyzer.start()


    def stop_baseline():
        time.sleep(30)
        session_data['baseline_mode'] = False
        session_data['baseline_captured'] = session_data['baseline_frame_count'] > 0
        if session_data['audio_baseline']:
            session_data['audio_baseline_stats'] = compute_audio_baseline(session_data['audio_baseline'])
        else:
            session_data['audio_baseline_stats'] = {}
        print(f"Baseline completed: frames={session_data['baseline_frame_count']}, captured={session_data['baseline_captured']}")

    
    threading.Thread(target=stop_baseline, daemon=True).start()
    
    return jsonify({'success': True, 'message': 'Baseline capture started (30 seconds)'})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start interrogation analysis"""
    if not session_data['active']:
        return jsonify({'success': False, 'error': 'No active session'})
    
    if session_data.get('baseline_mode', False):
        return jsonify({'success': False, 'error': 'Baseline in progress. Please wait until it completes.'})

    if not session_data.get('baseline_captured', False):
        return jsonify({'success': False, 'error': 'Baseline not captured'})
    
    session_data['analyzing'] = True
    session_data['emotion_log'] = []
    session_data['audio_log'] = []

    if not audio_analyzer.is_running():
        session_data['audio_available'] = audio_analyzer.start()

    return jsonify({'success': True, 'message': 'Analysis started'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop analysis"""
    session_data['analyzing'] = False
    session_data['audio_current'].pop('timestamp', None)
    return jsonify({'success': True, 'message': 'Analysis stopped'})

@app.route('/get_stats')
def get_stats():
    """Get real-time statistics"""
    if not session_data['active']:
        return jsonify({'error': 'No active session'})
    

    time_elapsed = "00:00:00"
    if session_data['start_time']:
        elapsed = datetime.now() - session_data['start_time']
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    

    emotion_stats = session_data.get('emotion_stats', {})
    top_emotions = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)[:3]

    if len(top_emotions) < 3:
        remaining = 3 - len(top_emotions)
        top_emotions.extend([('-', 0.0)] * remaining)


    stress_emotions = ['angry', 'fear', 'sad']
    stress_score = sum([emotion_stats.get(e, 0) for e in stress_emotions])
    if stress_score > 0.6:
        stress_level = "High"
    elif stress_score > 0.3:
        stress_level = "Moderate"
    else:
        stress_level = "Low"

    eye_insights = session_data.get('eye_insights', {})
    blink_count = eye_insights.get('blink_count', 0)
    blink_rate_per_min = 0.0
    if session_data['start_time'] and blink_count:
        elapsed_minutes = max((datetime.now() - session_data['start_time']).total_seconds() / 60.0, 0.1)
        blink_rate_per_min = round(blink_count / elapsed_minutes, 1)

    pose_insights = session_data.get('pose_insights', {})
    gesture_insights = session_data.get('gesture_insights', {})

    analyzer_summary = {
        'pose': {
            'lean_forward': pose_insights.get('lean_forward', False),
            'lean_backward': pose_insights.get('lean_backward', False),
            'shoulder_asymmetry': round(pose_insights.get('shoulder_asymmetry', 0.0), 2),
            'head_tilt_deg': round(pose_insights.get('head_tilt_deg', 0.0), 1)
        },
        'eye': {
            'blink_count': blink_count,
            'blink_rate_per_min': blink_rate_per_min,
            'eye_aspect_ratio': round(eye_insights.get('eye_aspect_ratio', 0.0), 3)
        },
        'gesture': {
            'hand_near_face': gesture_insights.get('hand_near_face', False),
            'hand_count': gesture_insights.get('hand_count', 0)
        }
    }

    audio_current = {
        k: sanitize_numeric(v) for k, v in session_data.get('audio_current', {}).items()
        if k != 'timestamp'
    }
    audio_status = {
        'available': bool(session_data.get('audio_available', False)),
        'current': audio_current,
        'baseline_stats': session_data.get('audio_baseline_stats', {}),
        'log_samples': len(session_data.get('audio_log', []))
    }

    return jsonify({
        'current_emotion': session_data['current_emotion'],
        'top_emotions': [{'name': e[0], 'value': round(e[1] * 100, 1)} for e in top_emotions],
        'confidence': 'High' if session_data.get('current_emotion_confidence', 0) > 0.7 else 'Moderate',
        'stress_level': stress_level,
        'microexpression_risk': 'Elevated' if stress_score > 0.5 else 'Normal',
        'total_logged': len(session_data['emotion_log']),
        'time_elapsed': time_elapsed,
        'baseline_captured': len(session_data['baseline_emotions']) > 0,
        'analyzers': analyzer_summary,
        'audio': audio_status
    })

@app.route('/get_emotion_timeline')
def get_emotion_timeline():
    """Get emotion timeline data for graph"""
    if not session_data['emotion_log']:
        return jsonify({'labels': [], 'datasets': []})
    
    df = pd.DataFrame(session_data['emotion_log'])
    

    df_sampled = df.iloc[::10]
    
    labels = [datetime.fromisoformat(t).strftime("%H:%M:%S") for t in df_sampled['timestamp']]
    
    datasets = []
    for emotion in ['happy', 'sad', 'fear']:
        if emotion in df_sampled.columns:
            datasets.append({
                'label': emotion.capitalize(),
                'data': (df_sampled[emotion] * 100).tolist()
            })
    
    return jsonify({'labels': labels, 'datasets': datasets})

@app.route('/save_report', methods=['POST'])
def save_report():
    """Save analysis report"""
    if not session_data['active'] or not session_data['emotion_log']:
        return jsonify({'success': False, 'error': 'No data to save'})
    

    df = pd.DataFrame(session_data['emotion_log'])
    os.makedirs('logs', exist_ok=True)
    filename = f"logs/{session_data['subject_id']}_{session_data['session_id']}.csv"
    df.to_csv(filename, index=False)
    
    files_saved = [filename]

    if session_data.get('audio_log'):
        audio_df = pd.DataFrame(session_data['audio_log'])
        audio_filename = f"logs/{session_data['subject_id']}_{session_data['session_id']}_audio.csv"
        audio_df.to_csv(audio_filename, index=False)
        files_saved.append(audio_filename)

    summary = {
        'subject_id': session_data['subject_id'],
        'session_id': session_data['session_id'],
        'total_records': len(df),
        'dominant_emotion': session_data['current_emotion'],
        'analysis_duration': session_data['start_time'].isoformat() if session_data['start_time'] else '',
        'stress_level': session_data.get('last_stress_level', 'Unknown'),
        'audio_baseline': session_data.get('audio_baseline_stats', {}),
        'audio_samples': len(session_data.get('audio_log', []))
    }
    
    summary_df = pd.DataFrame([summary])
    summary_filename = f"logs/{session_data['subject_id']}_{session_data['session_id']}_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    files_saved.append(summary_filename)
    
    return jsonify({'success': True, 'message': 'Report saved', 'files': files_saved})


@app.route('/assets/logo.png')
def serve_logo():
    return send_from_directory('logo', 'logo.png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6060, threaded=True)
