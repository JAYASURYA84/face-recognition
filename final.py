import face_recognition
import cv2

# Load the dataset
# Replace the following paths with the paths to your dataset and corresponding names
dataset_path = "D:\\face recognition\images"
names = ["jack","surya","dinesh","srk","sibhi"]  # Corresponding names for each person in the dataset

# Create arrays to store face encodings and corresponding names
known_face_encodings = []
known_face_names = []

for name in names:
    image = face_recognition.load_image_file(f"{dataset_path}/{name}.jpg")
    # Check if any faces are detected in the image
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        encoding = face_encodings[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the current face matches any face in the dataset
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

        name = "Unknown"

        # If a match is found, use the name of the first matching face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw the rectangle and name on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # If the recognized person is an authorized person, print a message
        if name in names:
            print(f"Authorized person: {name}")
            

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()


