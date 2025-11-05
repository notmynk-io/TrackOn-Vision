import cv2
from utils.camera_handler import CameraHandler
import time

def test_webcam():
    """
    Test webcam capture with threading.
    """
    print("=" * 50)
    print("Testing Webcam Capture")
    print("=" * 50)
    
    # camera handler for webcam
    camera = CameraHandler(camera_type="webcam")
    
    if not camera.initialize():
        print("[FAILED] Could not initialize webcam")
        return
    
    print("[SUCCESS] Webcam initialized!")
    print("[INFO] Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frame
            success, frame = camera.get_frame()
            
            if not success:
                print("[WARNING] Failed to get frame")
                continue
            
            frame_count += 1
            
            # Calculate and display FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Webcam Test - Press 'q' to quit", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:

        camera.release()
        cv2.destroyAllWindows()
        
        # Stats
        total_time = time.time() - start_time
        print(f"\n[STATS] Total frames: {frame_count}")
        print(f"[STATS] Total time: {total_time:.2f} seconds")
        print(f"[STATS] Average FPS: {frame_count / total_time:.2f}")


def test_ip_camera():
    """
    Test IP camera capture with threading.
    """
    print("=" * 50)
    print("Testing IP Camera Capture")
    print("=" * 50)
    
    # IP camera URL
    # RTSP: "rtsp://username:password@192.168.1.100:554/stream"
    # HTTP: "http://192.168.1.100:8080/video"
    ip_url = input("Enter your IP camera URL: ")
    
    # Initialize camera handler for IP camera
    camera = CameraHandler(camera_type="ip", ip_url=ip_url)
    
    if not camera.initialize():
        print("[FAILED] Could not initialize IP camera")
        return
    
    print("[SUCCESS] IP camera initialized!")
    print("[INFO] Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frame
            success, frame = camera.get_frame()
            
            if not success:
                print("[WARNING] Failed to get frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Calculate and display FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("IP Camera Test - Press 'q' to quit", frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        # stats
        total_time = time.time() - start_time
        print(f"\n[STATS] Total frames: {frame_count}")
        print(f"[STATS] Total time: {total_time:.2f} seconds")
        print(f"[STATS] Average FPS: {frame_count / total_time:.2f}")


if __name__ == "__main__":
    print("Select camera type to test:")
    print("1. Webcam")
    print("2. IP Camera")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        test_webcam()
    elif choice == "2":
        test_ip_camera()
    else:
        print("Invalid choice")
