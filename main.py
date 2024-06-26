import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

def take_picture():
    destination_window = "Camera view"
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("ERROR: Can't find a camera")
        return 1

    ret, frame = capture.read()
    count = 0

    print("Camera info")
    print("Width: " + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height: " + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS: " + str(capture.get(cv2.CAP_PROP_FPS)))
    print("Make a picture: ")
    print("\tPress 'q' for *.pgm (Grayscale)")
    print("\tPress 'w' for *.ppm (Colour)")
    print("\tPress 'e' for *.jpg (Colour)")
    print("\nPress 'ESC' to quit")

    cv2.namedWindow(destination_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(destination_window, frame)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("ERROR: Cannot receive images from camera.")
            return 1

        cv2.imshow(destination_window, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            filename = f"C:/RTT_Images/capture{count}.pgm"
            cv2.imwrite(filename, gray_image)
            count += 1
        elif key == ord('w'):
            filename = f"C:/RTT_Images/capture{count}.ppm"
            cv2.imwrite(filename, frame)
            count += 1
        elif key == ord('e'):
            filename = f"C:/RTT_Images/capture{count}.jpg"
            cv2.imwrite(filename, frame)
            count += 1
        elif key == 27: # ESC key
            print("Escape key pressed. Stopping picture mode.")
            break
        
    capture.release() # Release the camera
    cv2.destroyWindow("Camera view") # Close the window
    return 0

def record_video():
    destination_window = "Camera view"
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("ERROR: Can't find a camera")
        return 1

    print("Camera info")
    print("Width: " + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height: " + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS: " + str(capture.get(cv2.CAP_PROP_FPS)))
    print("Press 'r' to start recording.")
    print("Press 't' to stop recording.")
    print("Press 'ESC' to quit.")

    cv2.namedWindow(destination_window, cv2.WINDOW_AUTOSIZE)
    is_recording = False
    video_writer = None
    start_time = None
    elapsed_time = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("ERROR: Cannot receive images from camera.")
            break

        cv2.imshow(destination_window, frame)

        key = cv2.waitKey(1)
        if key == ord('r'):
            if not is_recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                save_path = os.path.join("C:/RTT_Videos", "output.avi")
                video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, 
                                (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                 int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                is_recording = True
                start_time = time.time()
                print("Recording started.")
        elif key == ord('t'):
            if is_recording:
                is_recording = False
                video_writer.release()
                print("Recording stopped.")
        elif key == 27 or elapsed_time >= 10: # ESC 
            if is_recording:
                is_recording = False
                video_writer.release()
            print("Exiting...")
            break

        if is_recording and video_writer is not None:
            video_writer.write(frame)
            elapsed_time = time.time() - start_time

    capture.release()
    cv2.destroyWindow("Camera view")
    return 0

            
def face_detectionWebcam():
    face_cascade = choose_cascade()

    destination_window = "Camera view"
    capture = cv2.VideoCapture(0)

    print("Camera info")
    print("Width: " + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height: " + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS: " + str(capture.get(cv2.CAP_PROP_FPS)))
    print("Press 'ESC' to quit")

    cv2.namedWindow(destination_window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(destination_window, 0, 0)

#___________________________________________________________________________________________________________________________________________________________
    image_dir = "C:/RTT_Images/" # Directory with face images
    face_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))] #

    match_scores = []  # List to store match scores
    faces_per_frame = []  # List to store the number of faces detected per frame
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("ERROR: No frame captured from camera.")
            return 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        frame_gray = cv2.equalizeHist(frame_gray) # Equalize histogram, improves face detection by improving contrast

        faces = face_cascade.detectMultiScale(frame_gray) 
        faces_per_frame.append(len(faces))  # Store the number of faces detected in this frame

        for (x, y, w, h) in faces: #x = x-coordinate, y = y-coordinate, w = width, h = height of detected face
            roi_gray = frame_gray[y:y+h, x:x+w] # Region of interest in grayscale: A rectangle in the grayscale frame where the face is detected. This is the frame used for template matching
            best_match_score = 0 
            best_match_image_name = None 

            for face_image_path in face_images:
                face_image = cv2.imread(face_image_path, cv2.IMREAD_GRAYSCALE) # Read face image in grayscale
                result = cv2.matchTemplate(roi_gray, face_image, cv2.TM_CCOEFF_NORMED) # Perform template matching between face image and detected face
                _, max_val, _, _ = cv2.minMaxLoc(result) #returns the maximum value of the result matrix. Matrix consists of max_value, min_value, max_location, min_location
                match_scores.append(max_val)  # Add match score to list
                
                if max_val > best_match_score:
                    best_match_score = max_val
                    best_match_image_name = os.path.basename(face_image_path) # Get filename of best match

            if best_match_score > 0.8:
                color = (0, 255, 0) # Green square if match score is high enough
                if best_match_image_name:
                    cv2.putText(frame, best_match_image_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)#Add text above square
            else:
                color = (0, 0, 255) # Red square if match score is too low

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2) # Draw rectangle around detected face

        cv2.imshow(destination_window, frame) 
        
#1. It first loads all the face images from a directory into the face_images list.
#2. It then enters an infinite loop where it continuously reads frames from a video capture.
#3. For each frame, it converts the frame to grayscale and equalizes the histogram to improve contrast.
#4. It then uses a face detection algorithm (face_cascade.detectMultiScale) to detect faces in the grayscale frame. This returns a list of rectangles where each rectangle represents a detected face.
#5. For each detected face, it extracts the region of interest (ROI) from the grayscale frame. The ROI is the part of the frame where the face is detected.
#6. It then loops through all the face images loaded earlier. For each face image, it performs template matching with the ROI using cv2.matchTemplate. This returns a result matrix.
#7. It finds the maximum value in the result matrix using cv2.minMaxLoc. This maximum value is the match score, which is a measure of how well the face image matches with the ROI.
#8. It keeps track of the face image with the highest match score.
#9. If the highest match score is above a certain threshold (0.8), it draws a green rectangle around the detected face in the original (not grayscale) frame and displays the name of the matching face image above the rectangle. If the match score is below the threshold, it draws a red rectangle.
#10. Finally, it displays the frame with the drawn rectangles and text.
#This process is repeated for each frame in the video capture, and for each detected face in each frame.
#___________________________________________________________________________________________________________________________________________________________

        if cv2.waitKey(10) == 27:
            print("Escape key pressed. Stopping face detection.")
            # Calculate percentage of matches above and below 0.8
            calculate_and_display_stats(match_scores, faces_per_frame, capture)
            return 0
        
def face_detectionVideo():
    face_cascade = choose_cascade()

    video_path = "C:/RTT_Videos/output.avi"  # Path to the saved video
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("ERROR: Can't open video file")
        return 1

    print("Video info")
    print("Width: " + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height: " + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS: " + str(capture.get(cv2.CAP_PROP_FPS)))
    print("Press 'ESC' to quit")

    destination_window = "Video view"
    cv2.namedWindow(destination_window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(destination_window, 0, 0)

    image_dir = "C:/RTT_Images/"  # Directory with face images
    face_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))] # List of face images

    match_scores = []  # List to store match scores
    faces_per_frame = []  # List to store the number of faces detected per frame

    while True:
        ret, frame = capture.read()
        if not ret:
            print("No more frames to read. Exiting.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame_gray = cv2.equalizeHist(frame_gray)  # Equalize histogram, improves face detection by improving contrast

        faces = face_cascade.detectMultiScale(frame_gray)
        faces_per_frame.append(len(faces))  # Store the number of faces detected in this frame

        for (x, y, w, h) in faces:
            roi_gray = frame_gray[y:y+h, x:x+w]  # Region of interest in grayscale
            best_match_score = 0
            best_match_image_name = None

            for face_image_path in face_images: # Loop through face images in directory and perform template matching
                face_image = cv2.imread(face_image_path, cv2.IMREAD_GRAYSCALE)  # Read face image in grayscale
                result = cv2.matchTemplate(roi_gray, face_image, cv2.TM_CCOEFF_NORMED)  # Perform template matching
                _, max_val, _, _ = cv2.minMaxLoc(result)  # Get maximum value of result matrix
                match_scores.append(max_val)  # Add match score to list

                if max_val > best_match_score: 
                    #The goal is to determine which face image from the directory matches best with the detected face in the video frame. 
                    # By comparing each face image and keeping track of the highest match score, 
                    # the code can identify the face image that is most similar to the detected face.
                    best_match_score = max_val
                    best_match_image_name = os.path.basename(face_image_path)  # Get filename of best match

            if best_match_score > 0.8: # Threshold for match score
                color = (0, 255, 0)  # Green square if match score is high enough
                if best_match_image_name:
                    cv2.putText(frame, best_match_image_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add text above square
            else:
                color = (0, 0, 255)  # Red square if match score is too low

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # Draw rectangle around detected face

        cv2.imshow(destination_window, frame)

        if cv2.waitKey(10) == 27:  # ESC key
            print("Escape key pressed. Stopping face detection.")
            break
    calculate_and_display_stats(match_scores, faces_per_frame, capture)
    #calculate_and_display_stats_saveGraph(match_scores, faces_per_frame, capture)
    return 0

def calculate_and_display_stats(match_scores, faces_per_frame, capture):
    # Calculate percentage of matches above and below 0.8
    above_threshold = sum(score > 0.8 for score in match_scores) # Count number of match scores above 0.8
    below_threshold = sum(score <= 0.8 for score in match_scores) # Count number of match scores below or equal to 0.8
    total_matches = len(match_scores) # Total number of match scores

    # Display percentage of matches
    above_percentage = (above_threshold / total_matches) * 100 # Calculate percentage of matches above 0.8
    below_percentage = (below_threshold / total_matches) * 100 # Calculate percentage of matches below or equal to 0.8
    print(f"Total number of matches: {total_matches}")
    print(f"Percentage of matches above 0.8: {above_percentage}%")
    print(f"Percentage of matches below 0.8: {below_percentage}%")

    # Display statistics for faces per frame
    avg_faces_per_frame = np.mean(faces_per_frame) # Calculate average number of faces detected per frame
    max_faces_per_frame = np.max(faces_per_frame) # Calculate maximum number of faces detected in a single frame
    print(f"Average number of faces per frame: {avg_faces_per_frame}")
    print(f"Maximum number of faces detected in a single frame: {max_faces_per_frame}")

    # Plot histogram of match scores
    plt.hist(match_scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Match Scores')

    # Plot histogram of faces per frame
    plt.figure()
    plt.hist(faces_per_frame, bins=range(1, max(faces_per_frame)+2), color='green', alpha=0.7, align='left')
    plt.xlabel('Number of Faces per Frame')
    plt.ylabel('Frequency')
    plt.title('Histogram of Faces Detected per Frame')

    capture.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close the windows
    plt.show()


def calculate_and_display_stats_saveGraph(match_scores, faces_per_frame, capture):
    # Specify the directory where you want to save the images
    save_directory = 'C:/RTT_Graphs'
    
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Calculate percentage of matches above and below 0.8
    above_threshold = sum(score > 0.8 for score in match_scores) # Count number of match scores above 0.8
    below_threshold = sum(score <= 0.8 for score in match_scores) # Count number of match scores below or equal to 0.8
    total_matches = len(match_scores) # Total number of match scores

    # Display percentage of matches
    above_percentage = (above_threshold / total_matches) * 100 # Calculate percentage of matches above 0.8
    below_percentage = (below_threshold / total_matches) * 100 # Calculate percentage of matches below or equal to 0.8
    print(f"Total number of matches: {total_matches}")
    print(f"Percentage of matches above 0.8: {above_percentage}%")
    print(f"Percentage of matches below 0.8: {below_percentage}%")

    # Display statistics for faces per frame
    avg_faces_per_frame = np.mean(faces_per_frame) # Calculate average number of faces detected per frame
    max_faces_per_frame = np.max(faces_per_frame) # Calculate maximum number of faces detected in a single frame
    print(f"Average number of faces per frame: {avg_faces_per_frame}")
    print(f"Maximum number of faces detected in a single frame: {max_faces_per_frame}")

    # Plot histogram of match scores
    plt.hist(match_scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Match Scores')
    match_scores_path = os.path.join(save_directory, 'match_scores_histogram.png')
    plt.savefig(match_scores_path) # Save the plot as an image
    plt.close() # Close the plot to prevent it from displaying

    # Plot histogram of faces per frame
    plt.figure()
    plt.hist(faces_per_frame, bins=range(1, max(faces_per_frame)+2), color='green', alpha=0.7, align='left')
    plt.xlabel('Number of Faces per Frame')
    plt.ylabel('Frequency')
    plt.title('Histogram of Faces Detected per Frame')
    faces_per_frame_path = os.path.join(save_directory, 'faces_per_frame_histogram.png')
    plt.savefig(faces_per_frame_path) # Save the plot as an image
    plt.close() # Close the plot to prevent it from displaying

    capture.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close the windows
    
    
def choose_cascade():
    #print("Choose a Haar cascade version:")
    #print("1: haarcascade_frontalface_default.xml")
    #print("2: haarcascade_frontalface_alt.xml")
    #print("3: haarcascade_frontalface_alt2.xml")
    #print("4: haarcascade_frontalface_alt_tree.xml")
    #optie = int(input())
    optie = 3
    
    if optie == 1:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    elif optie == 2:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    elif optie == 3:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    elif optie == 4:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

    if face_cascade.empty():
        print("ERROR: Cannot load face cascade.")
        return 1
    
    return face_cascade

def make_graph1():
    print("Aantal foto's (1 of 10): ")
    fotos = int(input())
    if fotos == 1:
        f = '1 foto'
    elif fotos == 10:
        f = '10 foto\'s'
        
# Data
    categories = ['F-L', 'F-U', 'B-L', 'B-U']
    TM_above_08 = [8.31,	13.20,	13.97,	8.66]
    TM_below_08 = [91.68,	86.80,	86.03,	91.34]


# Staafdiagram voor percentages boven en onder 0.8
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    index = range(len(categories))
    plt.bar(index, TM_above_08, bar_width, color='darkgrey', label='Boven 0.8')
    plt.bar([i + bar_width for i in index], TM_below_08, bar_width, color='grey', label='Onder 0.8')
    plt.xlabel('Categorieën')
    plt.ylabel('Percentage')
    plt.title('Percentage TM-waarden boven en onder 0.8 per categorie' + ' (' + f + ')')
    plt.xticks([i + bar_width / 2 for i in index], categories)
    plt.legend()
    plt.show()
    
    return 0
    
def make_graph2():
    print("Aantal foto's (1 of 10): ")
    fotos = int(input())
    if fotos == 1:
        f = '1 foto'
    elif fotos == 10:
        f = '10 foto\'s'
    
    print("Kies dataset (1 of 2): ")
    print("1: Variaties in Gezichtsbedekking")
    print("2: Variaties in Expressies")
    ds = int(input())
    
    if ds == 1:
        categories = ['F-B', 'F-P', 'F-M', 'B-B', 'B-P', 'B-M']
    elif ds == 2:
        categories = ['F-N', 'F-O', 'F-M', 'B-N', 'B-O', 'B-M']

    TM_above_08 = [13.20,	26.41,	3.30,	8.66,	10.28,	10.95]
    TM_below_08 = [86.80,	73.56,	96.70,	91.34,	89.72,	89.04]

    # Staafdiagram voor percentages boven en onder 0.8 (10 foto's)
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    index = range(len(categories))
    plt.bar(index, TM_above_08, bar_width, color='darkgrey', label='Boven 0.8')
    plt.bar([i + bar_width for i in index], TM_below_08, bar_width, color='grey', label='Onder 0.8')
    plt.xlabel('Categorieën')
    plt.ylabel('Percentage')
    plt.title('Percentage TM-waarden boven en onder 0.8 per categorie' + ' (' + f + ')')
    plt.xticks([i + bar_width / 2 for i in index], categories)
    plt.legend()
    plt.show()
    
    return 0

def main():
    while True:
        print("RTT Research  -  Make choice (give number + press Enter)")
        print("0: Close program")
        print("1: Take a picture")
        print("2: Record a video")
        print("3: Face detection (Webcam)")
        print("4: Face detection (Video)")
        print("5: Make graph for dataset 1")
        print("6: Make graph for dataset 2")
        optie = int(input())
        
        if optie == 0:
            print("Program closed.")
            return
        if optie == 1:
            ret = take_picture()
            if ret != 0:
                return ret
        if optie == 2:
            ret = record_video()
            if ret != 0:
                return ret
        elif optie == 3:
            ret = face_detectionWebcam()
            if ret != 0:
                return ret
        elif optie == 4:
            ret = face_detectionVideo()
            if ret != 0:
                return ret
        elif optie == 5:
            ret = make_graph1()
            if ret != 0:
                return ret
        elif optie == 6:
            ret = make_graph2()
            if ret != 0:
                return ret
        

if __name__ == "__main__":
    main()

#COMMANDS
#cd ..
#Desktop location: cd F:\Code\Python code\RTT
#.\FRS\Scripts\activate
#python main.py
#deactivate

# If virual environment won't start: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force