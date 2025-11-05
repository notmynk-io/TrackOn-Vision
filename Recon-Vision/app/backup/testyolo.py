# save as test_yolo.py
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Run prediction on a sample image
results = model('https://ultralytics.com/images/bus.jpg')

# Save each result (useful if multiple images/frames are passed)
for result in results:
    result.save()  # saves annotated image to default folder

print(results)
