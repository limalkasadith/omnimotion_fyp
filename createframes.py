import cv2
import os
import numpy as np
import sys

frame_list=[]
def split_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_number = 0
    

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        

        # Break the loop if there are no more frames
        if not ret or frame_number==60:
            break
        frame_list.append(frame)

        # Save the frame to the output folder
        frame_filename = f"{frame_number:05d}.png"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_number += 1
    
    # Release the video capture object
    cap.release()

# Example usage
if len(sys.argv) != 2:
    print("Usage: python createframes.py <video_path> ")
    sys.exit(1)
video_path = sys.argv[1]
output_folder = './jellyfish/color'
os.makedirs(output_folder, exist_ok=True)
split_frames(video_path, output_folder)
print("Frames saved successfully.")

def filter_images(frames_list, kernel,blue_lower,blue_upper):
    filtered_frames = []
    for frame in frames_list:
        image2=np.zeros(frame.shape,dtype=np.uint8)
        blue_mask = (frame[:, :, 0] >= blue_lower) & (frame[:, :, 0] <= blue_upper)
        image2[blue_mask] = [0, 0, 255]
        filtered_frames.append(cv2.filter2D(image2, -1, kernel))
       
    return filtered_frames

blue_lower = 90  # Adjust this range as needed
blue_upper = 220  # Adjust this range as needed
kernel = np.ones((5, 5), np.uint8)
filtered_frames = filter_images(frame_list, kernel,blue_lower,blue_upper)


def save_frames_with_leading_zeros(frames, output_folder):
    for i, frame in enumerate(frames):
        # Define the filename with leading zeros
        filename = f"{output_folder}/{i:05d}.png"
       
        cv2.imwrite(filename, frame)

# Example usage:
# Assuming frames_list is the list of frames obtained earlier
output_folder = './jellyfish/mask'  # Create a folder to save the frames
os.makedirs(output_folder, exist_ok=True)
# Save frames with leading zeros
save_frames_with_leading_zeros(filtered_frames, output_folder)
print("Mask saved successfully.")

