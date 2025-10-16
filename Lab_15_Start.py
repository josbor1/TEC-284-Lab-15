import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Initialize the Picamera2
def initialize_camera():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1000, 800)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    return picam2

# Load the YOLO model
def load_model(model_path="yolo11n.pt"):
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Main processing loop
def run_inference(picam2, model):
    fps_start_time = time.time()

    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Flip the frame vertically. "0" is the angle, so alter
        # this as needed to get your camera oriented properly
        flipped_frame = cv2.flip(frame, 0)

        # Run inference with YOLO
        results = model(flipped_frame)

        # Annotate the frame with the results
        annotated_frame = results[0].plot()

        # Get inference speed and calculate FPS
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time  # Convert to milliseconds
        text = f'FPS: {fps:.1f}'

        # Define font and position for FPS text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10

        # Draw FPS on the frame
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("Camera", annotated_frame)

        # Handle exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        # Update FPS counter
        if time.time() - fps_start_time >= 1.0:
            fps_start_time = time.time()

# Main function to initialize and run
def main():
    picam2 = initialize_camera()
    model = load_model("yolo11n.pt")

    if model is not None:
        try:
            run_inference(picam2, model)
        except KeyboardInterrupt:
            print("Process interrupted by the user.")
    else:
        print("Failed to load the YOLO model. Exiting.")

    # Cleanup: Release resources when done
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()

