import cv2
from image_processing import apply_processing
import motion_detection


def process_video_interactive():
    """
    Captures video from the webcam and allows real-time processing based on user key inputs.

    Key Controls:
    - ESC: Exit the program.
    - 0: Reset processing to the original frame.
    - 1, 2, 3, 5, 6, 7, 8, r, t, y, f, c, s, b, n, g, h, e: Apply different transformations.
    - 4: Switch to motion detection using the default method.
    - 9: Switch to motion detection using the frame difference method (with grayscale).
    """
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_key = None  # Store the last key pressed

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # Exit if the frame could not be read

        # Apply the last selected transformation if a valid key was pressed
        if last_key:  # Apply the last selected transformation
            processed = apply_processing(frame, last_key)
        else:
            processed = frame.copy()  # Default to showing the original frame
        # Display the original and processed video feed
        cv2.imshow("Original Video", frame)
        cv2.imshow("Processed Video", processed)
        # Capture user key press (wait for 1ms per frame and read the pressed key)
        key = cv2.waitKey(1) & 0xFF  # Check for key press
        if key == 27:  # ESC to exit
            break
        elif key == ord('0'):  # Reset processing
            last_key = None
        elif key in [ord(k) for k in '1235678rtyfcsbnghe']:  # Only update valid keys
            last_key = key  # Store the last valid key
        elif key == ord('4'):  # Only update valid keys
            cv2.destroyAllWindows()
            motion_detection.motion_detection(cap)
        elif key == ord('9'):  # Only update valid keys
            cv2.destroyAllWindows()
            motion_detection.motion_detection_frame_difference(cap, grayscale=True)
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
