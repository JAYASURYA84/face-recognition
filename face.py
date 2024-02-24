import cv2
import os

# Create a directory to store the dataset
dataset_path = "D:\\face recognition\images"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam (you may need to change the index based on your system)
cap = cv2.VideoCapture(0)

# Counter for the number of captured images
image_count = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces2
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Capture Faces', frame)

    # Capture the face if 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Increment the image count
        image_count += 1

        # Save the captured face to the dataset directory
        face_filename = f'person_{image_count}.jpg'
        face_path = os.path.join(dataset_path, face_filename)
        cv2.imwrite(face_path, frame[y:y+h, x:x+w])

        print(f'Face captured and saved as {face_filename}')

    # Break the loop if 'q' is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
