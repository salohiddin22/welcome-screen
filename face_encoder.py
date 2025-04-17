import os
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

def encode_faces():
    """
    Encodes all employee faces from the employee_images directory and saves to encodings.p
    """
    # Initialize Firebase (database only)
    if not firebase_admin._apps:
        cred = credentials.Certificate("service_account_key.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://welcome-screen-9366d-default-rtdb.firebaseio.com/'
        })
    
    # Reference to Firebase database
    ref = db.reference('Employees')
    
    # Get all employee IDs from Firebase
    all_employees = ref.get()
    if not all_employees:
        print("No employees found in database!")
        print("Please run firebase_setup_local.py first to set up the database.")
        return
    
    print(f"Found {len(all_employees)} employees in database")
    
    # Path to employee images
    path = 'employee_images'
    
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        print("Please add employee images to this directory")
        return
    
    # Process each image in the directory
    images = []
    employee_ids = []
    employee_names = []
    
    for img_name in os.listdir(path):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            employee_id = img_name.split('.')[0]  # Remove file extension
            
            # Check if this ID exists in Firebase
            employee_data = ref.child(employee_id).get()
            if not employee_data:
                print(f"Warning: {employee_id} not found in database, skipping...")
                continue
                
            # Read the image
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                print(f"Error: Could not read image {img_name}")
                continue
                
            images.append(img)
            employee_ids.append(employee_id)
            employee_names.append(employee_data.get('name', 'Unknown'))
            print(f"Added {employee_id}: {employee_data.get('name', 'Unknown')} to encoding list")
    
    if not images:
        print("No valid images found in directory!")
        return
        
    print(f"Found {len(images)} images. Starting encoding process...")
    
    # Function to find encodings
    def find_encodings(images_list):
        encode_list = []
        for idx, img in enumerate(images_list):
            # Convert to RGB as face_recognition uses RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            try:
                print(f"Processing image for {employee_ids[idx]}...")
                face_locations = face_recognition.face_locations(img)
                if not face_locations:
                    print(f"Warning: No face detected in image for {employee_ids[idx]}")
                    encode_list.append(None)
                    continue
                    
                encode = face_recognition.face_encodings(img, face_locations)[0]
                encode_list.append(encode)
                print(f"Successfully encoded face for {employee_ids[idx]}")
            except Exception as e:
                print(f"Error encoding image for {employee_ids[idx]}: {str(e)}")
                encode_list.append(None)
        
        return encode_list
    
    # Encode images
    print("Encoding images...")
    encode_list_known = find_encodings(images)
    
    # Filter out None values (failed encodings)
    valid_encodings = []
    valid_ids = []
    
    for encode, emp_id in zip(encode_list_known, employee_ids):
        if encode is not None:
            valid_encodings.append(encode)
            valid_ids.append(emp_id)
    
    encode_list_known_with_ids = [valid_encodings, valid_ids]
    
    print(f"Encoding Complete. Successfully encoded {len(valid_encodings)} faces out of {len(images)} images.")
    
    # Save encodings to file
    print("Saving encodings to file...")
    with open('encodings.p', 'wb') as file:
        pickle.dump(encode_list_known_with_ids, file)
    
    print(f"Encodings saved to encodings.p")
    print("Encoded employees:")
    for idx, emp_id in enumerate(valid_ids):
        print(f"  - {emp_id}: {ref.child(emp_id).child('name').get()}")

def debug_face_detection():
    """
    Debug helper to test if face detection is working with your images
    """
    path = 'employee_images'
    
    if not os.path.exists(path):
        print("Employee images directory not found!")
        return
    
    for img_name in os.listdir(path):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            print(f"Testing image: {img_name}")
            img = cv2.imread(os.path.join(path, img_name))
            if img is None:
                print(f"  Error: Could not read image {img_name}")
                continue
                
            # Convert to RGB for face_recognition
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                print(f"  WARNING: No faces detected in {img_name}")
                print("  Try a clearer image with a front-facing person")
            else:
                print(f"  SUCCESS: Detected {len(face_locations)} face(s) in {img_name}")
                
                # Draw boxes around faces and save debug image
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                debug_dir = 'debug_faces'
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                    
                cv2.imwrite(os.path.join(debug_dir, f"debug_{img_name}"), img)
                print(f"  Debug image saved to debug_faces/debug_{img_name}")
            
            print("")
    
    print("Face detection test complete. Check the debug_faces folder for results.")

if __name__ == "__main__":
    print("Face Encoder - Local Storage Version")
    print("===================================")
    print("This script will encode employee faces from local storage.")
    print("Choose an option:")
    print("1. Encode faces")
    print("2. Debug face detection (tests your images)")
    
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        encode_faces()
    elif choice == "2":
        debug_face_detection()
    else:
        print("Invalid choice. Exiting.")