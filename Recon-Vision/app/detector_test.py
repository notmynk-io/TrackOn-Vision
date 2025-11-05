import cv2
from app.detector1 import Detector

def main():
    detector = Detector(device='cpu', conf_threshold=0.3)

    # Load an image from local file - update path as needed
    img = cv2.imread('data/images.jpeg')

    if img is None:
        raise FileNotFoundError("Image 'data/images.jpeg' not found or unreadable.")

    detections, annotated_frame = detector.detect(img)

    # Print detections
    print("Detections:")
    for d in detections:
        print(d)

    # Show annotated frame
    cv2.imshow('Detection', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save annotated image
    cv2.imwrite('app/bus_annotated.jpg', annotated_frame)

if __name__ == '__main__':
    main()