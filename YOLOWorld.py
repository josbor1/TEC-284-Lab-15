import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Initialize the Picamera2
def initialize_camera(resolution=(640, 480)):
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = resolution
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    return picam2

# Load the YOLO World model
def load_model(model_path="yolov8n-world.pt", classes=["person"]):
    try:
        model = YOLO(model_path)
        model.set_classes(classes)
        print(f"YOLO World model loaded with classes: {classes}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Main processing loop
def run_inference(picam2, model, display=True):
    print("Starting inference loop. Press 'q' to quit.")
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Run inference with streaming
        results = model.predict(frame, stream=True)

        for result in results:
            # Annotate the frame with the detection results
            annotated_frame = result.plot()

            # Calculate FPS from inference speed (ms)
            inference_time = result.speed['inference']
            fps = 1000 / inference_time if inference_time > 0 else 0
            fps_text = f'FPS: {fps:.1f}'

            # Overlay FPS on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fps_text, font, 1, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(annotated_frame, fps_text, (text_x, text_y),
                        font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the frame if display is enabled
            if display:
                cv2.imshow("YOLO World - Camera", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

# Main function
def main():
    picam2 = initialize_camera(resolution=(640, 480))
    model = load_model("yolov8s-world.pt", classes=["person"])

    if model is not None:
        try:
            run_inference(picam2, model, display=True)  # Set display=False for better speed
        except KeyboardInterrupt:
            print("Interrupted by user.")
    else:
        print("Model failed to load.")

    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    print("Resources released. Exiting.")

if __name__ == "__main__":
    main()
