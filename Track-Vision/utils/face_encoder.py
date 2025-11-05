import face_recognition
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

class FaceEncoder:
    """
    Handles face detection and encoding extraction from images.
    """
    
    def __init__(self, model: str = "hog"):
        """
        Initialize the FaceEncoder.
        
        Args:
            model: Detection model to use - 'hog' (faster, CPU) or 'cnn' (accurate, GPU)
        """
        self.model = model
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for better face detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of preprocessed image and the scale factor applied relative to
            the original dimensions.
        """

        height, width = image.shape[:2]
        max_dimension = 800
        scale = 1.0
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        if len(image.shape) == 3:

            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return image, scale
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = face_recognition.load_image_file(image_path)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect face locations in the image.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        face_locations = face_recognition.face_locations(image, model=self.model)
        return face_locations
    
    def encode_faces(self, image: np.ndarray, 
                     face_locations: Optional[List[Tuple]] = None) -> List[np.ndarray]:
        """
        Generate 128-dimensional face encodings.
        
        Args:
            image: Input image as numpy array (RGB format)
            face_locations: Pre-detected face locations (optional)
            
        Returns:
            List of face encodings (128-d vectors)
        """
        encodings = face_recognition.face_encodings(
            image, 
            known_face_locations=face_locations
        )
        return encodings
    
    def process_uploaded_photo(self, image_path: str) -> Tuple[Optional[np.ndarray], 
                                                                 Optional[List], 
                                                                 Optional[np.ndarray]]:
        """
        Complete pipeline: Load image, detect face, and extract encoding.
        
        Args:
            image_path: Path to the uploaded photo
            
        Returns:
            Tuple of (original_image, face_locations, face_encoding)
            Returns (None, None, None) if processing fails
        """
        print(f"[INFO] Processing uploaded photo: {image_path}")
        
        image = self.load_image(image_path)
        if image is None:
            print("[ERROR] Failed to load image")
            return None, None, None
        
        print(f"[INFO] Image loaded successfully. Shape: {image.shape}")
        
        preprocessed, scale = self.preprocess_image(image.copy())
        
        print("[INFO] Detecting faces...")
        face_locations = self.detect_faces(preprocessed)
        
        if len(face_locations) == 0:
            print("[WARNING] No faces detected in the image")
            return image, [], None
        
        print(f"[INFO] Found {len(face_locations)} face(s)")
        
        encode_locations = face_locations
        if len(encode_locations) > 1:
            print("[WARNING] Multiple faces detected. Using the largest face.")
            face_areas = [(bottom - top) * (right - left) 
                         for (top, right, bottom, left) in encode_locations]
            largest_face_idx = np.argmax(face_areas)
            encode_locations = [encode_locations[largest_face_idx]]
        

        print("[INFO] Generating face encoding...")
        encodings = self.encode_faces(preprocessed, encode_locations)
        
        if len(encodings) == 0:
            print("[ERROR] Failed to generate face encoding")
            return image, encode_locations, None
        
        face_encoding = encodings[0]
        print(f"[INFO] Face encoding generated successfully. Shape: {face_encoding.shape}")
        
        if scale != 1.0:
            img_height, img_width = image.shape[:2]
            adjusted_locations = []
            for top, right, bottom, left in encode_locations:
                adj_top = max(0, min(img_height, int(round(top / scale))))
                adj_right = max(0, min(img_width, int(round(right / scale))))
                adj_bottom = max(0, min(img_height, int(round(bottom / scale))))
                adj_left = max(0, min(img_width, int(round(left / scale))))
                adjusted_locations.append((adj_top, adj_right, adj_bottom, adj_left))
            face_locations = adjusted_locations
        else:
            face_locations = encode_locations
        
        return image, face_locations, face_encoding
    
    def save_encoding(self, encoding: np.ndarray, save_path: str) -> bool:
        """
        Save face encoding to disk.
        
        Args:
            encoding: Face encoding array
            save_path: Path to save the encoding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            np.save(save_path, encoding)
            print(f"[INFO] Encoding saved to {save_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save encoding: {e}")
            return False
    
    def load_encoding(self, encoding_path: str) -> Optional[np.ndarray]:
        """
        Load face encoding from disk.
        
        Args:
            encoding_path: Path to the encoding file
            
        Returns:
            Face encoding array or None if failed
        """
        try:
            encoding = np.load(encoding_path)
            print(f"[INFO] Encoding loaded from {encoding_path}")
            return encoding
        except Exception as e:
            print(f"[ERROR] Failed to load encoding: {e}")
            return None
