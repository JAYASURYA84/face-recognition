import face_recognition
import os

# Load the known faces and their corresponding labels from the dataset
def load_known_faces(dataset_path):
    known_faces = []
    labels = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.jpg'):
            # Load the image
            image_path = os.path.join(dataset_path, file_name)
            face_image = face_recognition.load_image_file(image_path)

            # Get face encodings (if any)
            face_encodings = face_recognition.face_encodings(face_image)

            if face_encodings:
                # Take the first face encoding
                face_encoding = face_encodings[0]

                # Extract label from the filename (assuming the filename is in the format "person_{label}.jpg")
                label = int(file_name.split('_')[1].split('.')[0])

                known_faces.append(face_encoding)
                labels.append(label)
            else:
                print(f"No face found in {image_path}")

    return known_faces, labels

# Authenticate a person based on a new face
def authenticate_person(new_face_image, known_faces, labels):
    # Encode the new face
    new_face_encoding = face_recognition.face_encodings(new_face_image)[0]

    # Compare with known faces
    matches = face_recognition.compare_faces(known_faces, new_face_encoding)

    if True in matches:
        # Identify the label of the recognized face
        index = matches.index(True)
        label = labels[index]
        return label
    else:
        return None

# Specify the path to your dataset
dataset_path = "D:\\face recognition\\images"

# Load known faces and labels
known_faces, labels = load_known_faces(dataset_path)

# Capture a new face (replace with the path to your new face image)
new_face_image_path ="D:\face recognition\newimages\newface.jpeg"  # Replace with the path to your new face image
new_face_image = face_recognition.load_image_file(new_face_image_path)

# Authenticate the person
authenticated_label = authenticate_person(new_face_image, known_faces, labels)

if authenticated_label is not None:
    print(f"Person with label {authenticated_label} is authorized.")
    
    # Check if the person is already in the dataset
    if authenticated_label in labels:
        print("Person is already in the dataset.")
    else:
        print("Person is not in the dataset.")
else:
    print("Unauthorized person.")



