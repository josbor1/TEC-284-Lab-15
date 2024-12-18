import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gpiozero import LED

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1000, 800)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# initialise output pin. You will need to change this to whatever number you are using!
output = LED(16)

# Load the YOLO model
model = YOLO("yolo11n_ncnn_model")

# List of class IDs we want to detect
objects_to_detect = [0, 73]  # You can modify this list. 0 is a human and 73 is a book. See below for other examples:
'''
{0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'} '''

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Flip the frame vertically
    flipped_frame = cv2.flip(frame, 0)
    
    # Run YOLO model on the flipped frame and store the results
    results = model.predict(flipped_frame, imgsz = 320)

    # Get the classes of detected objects
    detected_objects = results[0].boxes.cls.tolist()

    # Check if any of our specified objects are detected
    object_found = False
    for obj_id in objects_to_detect:
        if obj_id in detected_objects:
            object_found = True
            print(f"Detected object with ID {obj_id}!")
    
    # Control the Pin based on detection
    if object_found:
        output.on()  # Turn on Pin
        print("Pin turned on!")
    else:
        output.off()   # Turn off Pin
        print("Pin turned off!")
            
    # Display the frame with detection results
    annotated_frame = results[0].plot()
    cv2.imshow("Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
