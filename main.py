import image_processing
from video_processing import process_video_interactive


def main():
    """
    Main function to allow the user to select between image processing and video processing.

    The user is prompted to choose between processing an image or a video.
    Depending on the selection, the appropriate processing function is called.
    """
    while 1:
        mode = input("please select [0] for image  [1] for video: ")
        # Validate user input to ensure it's either '0' or '1'
        if mode.isdigit() and int(mode) in (0, 1):
            break
        else:
            print("please enter a valid option")
    # Call the appropriate function based on user selection
    if int(mode):
        process_video_interactive()  # Process video interactively
    else:
        image_processing.file_selector()  # Open file selector for image processing
# Entry point of the script


if __name__ == "__main__":
    main()
