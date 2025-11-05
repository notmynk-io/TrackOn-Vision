import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
from concurrent.futures import ThreadPoolExecutor
import threading

class FaceMatcher:
    """
    Real-time face matching engine with parallel processing.
    """
    
    def __init__(self, reference_encoding: np.ndarray, 
                 tolerance: float = 0.6,
                 detection_model: str = "hog",
                 enable_parallel: bool = True):
        """
        Initialize the face matcher.
        
        Args:
            reference_encoding: 128-d face encoding of the target person
            tolerance: Match threshold (0.6 default, lower = stricter)
            detection_model: 'hog' (fast, CPU) or 'cnn' (accurate, GPU)
            enable_parallel: Enable parallel processing for multiple faces
        """
        self.reference_encoding = reference_encoding
        self.tolerance = tolerance
        self.detection_model = detection_model
        self.enable_parallel = enable_parallel
        
        # Performance
        self.total_frames = 0
        self.matched_frames = 0
        self.processing_times = []
        
        # for parallel processing
        if enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=4)
        else:
            self.executor = None
        
        print(f"[INFO] Face Matcher initialized")
        print(f"[INFO] Tolerance: {tolerance} (lower = stricter)")
        print(f"[INFO] Detection model: {detection_model}")
        print(f"[INFO] Parallel processing: {enable_parallel}")
    
    def detect_and_encode_faces(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces and generate encodings in a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (face_locations, face_encodings)
        """
        # Convert BGR to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame (faster processing) 
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(
            small_frame, 
            model=self.detection_model
        )
        
        # restore face locations
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                         for (top, right, bottom, left) in face_locations]
        
        # encodings for detected faces
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations
            )
        else:
            face_encodings = []
        
        return face_locations, face_encodings
    
    def compare_face(self, face_encoding: np.ndarray) -> Tuple[bool, float]:
        """
        Compare a single face encoding with reference encoding.
        
        Args:
            face_encoding: Face encoding to compare
            
        Returns:
            Tuple of (is_match, distance)
        """
        # Calculate face distance
        distance = face_recognition.face_distance(
            [self.reference_encoding], 
            face_encoding
        )[0]
        
        # Check if it's a match
        is_match = distance <= self.tolerance
        
        return is_match, distance
    
    def compare_multiple_faces(self, face_encodings: List[np.ndarray]) -> List[Dict]:
        """
        Compare multiple face encodings with reference (parallel processing).
        
        Args:
            face_encodings: List of face encodings to compare
            
        Returns:
            List of match results with confidence scores
        """
        results = []
        
        if self.enable_parallel and len(face_encodings) > 1:
            # Parallel processing for multiple faces
            futures = [self.executor.submit(self.compare_face, encoding) 
                      for encoding in face_encodings]
            
            for future in futures:
                is_match, distance = future.result()
                confidence = max(0, (1 - distance) * 100)  # Convert to percentage
                results.append({
                    'is_match': is_match,
                    'distance': distance,
                    'confidence': confidence
                })
        else:
            # Sequential processing
            for encoding in face_encodings:
                is_match, distance = self.compare_face(encoding)
                confidence = max(0, (1 - distance) * 100)
                results.append({
                    'is_match': is_match,
                    'distance': distance,
                    'confidence': confidence
                })
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect, encode, and match faces.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, match_results)
        """
        start_time = time.time()
        
        face_locations, face_encodings = self.detect_and_encode_faces(frame)
        
        # Compare with uploaded images
        if len(face_encodings) > 0:
            match_results = self.compare_multiple_faces(face_encodings)
        else:
            match_results = []
        
        annotated_frame = self.annotate_frame(
            frame, 
            face_locations, 
            match_results
        )
        
        # Track stats perfor
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.total_frames += 1
        
        # Count matched frames
        if any(result['is_match'] for result in match_results):
            self.matched_frames += 1
        
        return annotated_frame, match_results
    
    def annotate_frame(self, frame: np.ndarray, 
                      face_locations: List[Tuple], 
                      match_results: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on detected faces.
        
        Args:
            frame: Input frame
            face_locations: List of face locations
            match_results: List of match results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for (top, right, bottom, left), result in zip(face_locations, match_results):
            # Determine color based on match
            if result['is_match']:
                color = (0, 255, 0)  # Green for match
                label = f"MATCH ({result['confidence']:.1f}%)"
            else:
                color = (0, 0, 255)  # Red for no match
                label = f"NOT MATCH ({result['confidence']:.1f}%)"
            
            # Draw rectangle
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(annotated, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(annotated, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw distance info
            distance_text = f"Dist: {result['distance']:.3f}"
            cv2.putText(annotated, distance_text, (left + 6, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def get_statistics(self) -> Dict:
        """
        Get matching statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        avg_processing_time = (np.mean(self.processing_times) 
                              if self.processing_times else 0)
        
        match_rate = (self.matched_frames / self.total_frames * 100 
                     if self.total_frames > 0 else 0)
        
        return {
            'total_frames': self.total_frames,
            'matched_frames': self.matched_frames,
            'match_rate': match_rate,
            'avg_processing_time': avg_processing_time,
            'fps': 1 / avg_processing_time if avg_processing_time > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset performance tracking statistics."""
        self.total_frames = 0
        self.matched_frames = 0
        self.processing_times = []
    
    def shutdown(self):
        """Shutdown the matcher and release resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        print("[INFO] Face Matcher shutdown complete")
