import cv2
from utils.face_encoder import FaceEncoder
from pathlib import Path

def test_face_encoding():
    """
    Test the face encoding functionality.
    """
    encoder = FaceEncoder(model="cnn")
    
  
    test_image_path = "uploaded_photos/person.jpg"
    

    image, face_locations, face_encoding = encoder.process_uploaded_photo(test_image_path)
    
    if face_encoding is None:
        print("\n[FAILED] Could not extract face encoding")
        return
    
    print("\n[SUCCESS] Face encoding extracted!")
    print(f"Encoding shape: {face_encoding.shape}")
    print(f"Encoding sample (first 10 values): {face_encoding[:10]}")
    

    if len(face_locations) > 0:

        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display_image, "Detected Face", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        

        cv2.imshow("Face Detection Result", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    encoding_path = "uploaded_photos/person_encoding.npy"
    encoder.save_encoding(face_encoding, encoding_path)
    

    loaded_encoding = encoder.load_encoding(encoding_path)
    print(f"\n[INFO] Encoding loaded successfully: {loaded_encoding is not None}")

if __name__ == "__main__":

    Path("uploaded_photos").mkdir(exist_ok=True)
    
    test_face_encoding()
