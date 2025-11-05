import cv2
import numpy as np
from utils.face_encoder import FaceEncoder
from utils.camera_handler import CameraHandler
from utils.face_matcher import FaceMatcher
import time
import sys

def main():
    """
    Main application for real-time face recognition.
    """
    print("=" * 60)
    print("FACERECON - Real-Time Face Recognition System")
    print("=" * 60)
    
    # Loading reference image and extract encoding
    print("\n[STEP 1] Loading reference photo...")
    reference_image_path = input("Enter path to reference photo: ")
    
    encoder = FaceEncoder(model="hog")
    image, face_locations, face_encoding = encoder.process_uploaded_photo(
        reference_image_path
    )
    
    if face_encoding is None:
        print("[ERROR] Could not extract face from reference photo")
        print("[ERROR] Please ensure the photo contains a clear, visible face")
        sys.exit(1)
    
    print("[SUCCESS] Reference face encoding extracted!")
    
    # Save encoding for future use
    save_encoding = input("\nSave encoding for future use? (y/n): ")
    if save_encoding.lower() == 'y':
        encoding_path = "uploaded_photos/reference_encoding.npy"
        encoder.save_encoding(face_encoding, encoding_path)
    
    # Setup camera (Webcam or IP Camera)
    print("\n[STEP 2] Setting up camera...")
    camera_type = input("Select camera type (1=Webcam, 2=IP Camera): ")
    
    if camera_type == "1":
        camera = CameraHandler(camera_type="webcam")
    elif camera_type == "2":
        ip_url = input("Enter IP camera URL: ")
        camera = CameraHandler(camera_type="ip", ip_url=ip_url)
    else:
        print("[ERROR] Invalid camera type")
        sys.exit(1)
    
    if not camera.initialize():
        print("[ERROR] Failed to initialize camera")
        sys.exit(1)
    
    print("[SUCCESS] Camera initialized!")
    
    # Face matcher
    print("\n[STEP 3] Setting up face matcher...")
    tolerance = input("Enter tolerance (0.4-0.7, default 0.6, lower=stricter): ")
    
    try:
        tolerance = float(tolerance) if tolerance else 0.6
    except ValueError:
        tolerance = 0.6
    
    matcher = FaceMatcher(
        reference_encoding=face_encoding,
        tolerance=tolerance,
        detection_model="hog",
        enable_parallel=True
    )
    
    print("[SUCCESS] Face matcher initialized!")
    
    # Start real-time recognition
    print("\n[STEP 4] Starting real-time face recognition...")
    print("[INFO] Press 'q' to quit, 's' to show statistics")
    print("-" * 60)
    
    try:
        frame_count = 0
        fps_start_time = time.time()
        display_fps = 0
        
        while True:
            # Get frame from camera
            success, frame = camera.get_frame()
            
            if not success:
                print("[WARNING] Failed to get frame")
                time.sleep(0.1)
                continue
            
            # Process frames for face matching
            annotated_frame, match_results = matcher.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                display_fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
            
            # Add FPS to display
            cv2.putText(annotated_frame, f"FPS: {display_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frames: {frame_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255), 2)
            
            # Display match status
            if any(result['is_match'] for result in match_results):
                status_text = "TARGET FOUND!"
                status_color = (0, 255, 0)
            elif len(match_results) > 0:
                status_text = "Faces detected (no match)"
                status_color = (0, 165, 255)
            else:
                status_text = "Searching..."
                status_color = (255, 255, 255)
            
            cv2.putText(annotated_frame, status_text, 
                       (10, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Show frame
            cv2.imshow("FACERECON - Real-Time Face Recognition", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                # Show statistics
                stats = matcher.get_statistics()
                print("\n" + "=" * 60)
                print("STATISTICS")
                print("=" * 60)
                print(f"Total frames processed: {stats['total_frames']}")
                print(f"Frames with match: {stats['matched_frames']}")
                print(f"Match rate: {stats['match_rate']:.2f}%")
                print(f"Avg processing time: {stats['avg_processing_time']*1000:.2f}ms")
                print(f"Processing FPS: {stats['fps']:.2f}")
                print("=" * 60 + "\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        print("\n[CLEANUP] Releasing resources...")
        camera.release()
        matcher.shutdown()
        cv2.destroyAllWindows()
        
        # Final statistics
        stats = matcher.get_statistics()
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Frames with match: {stats['matched_frames']}")
        print(f"Match rate: {stats['match_rate']:.2f}%")
        print(f"Avg processing time: {stats['avg_processing_time']*1000:.2f}ms")
        print("=" * 60)
        
        print("\n[INFO] Application closed successfully")

if __name__ == "__main__":
    main()
