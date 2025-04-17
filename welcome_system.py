import os
import pickle
import time
import random
import json
import threading
import requests
from datetime import datetime
from queue import Queue

import cv2
import cvzone
import face_recognition
import numpy as np
import pygame
from gtts import gTTS

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase (database only)
cred = credentials.Certificate("service_account_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://welcome-screen-9366d-default-rtdb.firebaseio.com/'
})

 #Initialize pygame for audio playback
pygame.mixer.init()

# Initialize face detection and recognition
cap = cv2.VideoCapture(2)
cap.set(3, 1280)  # Higher resolution for better detection
cap.set(4, 720)

# Load the background for welcome screen
imgBackground = cv2.imread('resources/cover.jpg')

# Load the encoding file
print('Loading encoding file...')
file = open('encodings.p', 'rb')
encode_list_known_with_ids = pickle.load(file)
file.close()
encode_list_known, employee_ids = encode_list_known_with_ids
print('Encoding file loaded')

# Load slideshow images
slideshow_dir = 'slideshow_images'
slideshow_images = [os.path.join(slideshow_dir, img) for img in os.listdir(slideshow_dir) if img.endswith(('.jpg', '.png'))]
current_slideshow_index = 0

# Initialize first slideshow image
if slideshow_images:
    slideshow_img = cv2.imread(slideshow_images[current_slideshow_index])
    slideshow_img = cv2.resize(slideshow_img, (1080, 1920))
else:
    # Create a blank image if no slideshow images are available
    print("Warning: No slideshow images found. Creating blank slideshow.")
    slideshow_img = np.ones((1920, 1080, 3), dtype=np.uint8) * 240  # Light gray
    cv2.putText(slideshow_img, "No Slideshow Images Available", (200, 960), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

# Local image directory
employee_images_dir = "employee_images"

# Audio queue for background processing
audio_queue = Queue()

# Constants
RECOGNITION_TIMEOUT = 10  # seconds
SLIDESHOW_INTERVAL = 5    # seconds
FACE_CONFIDENCE_THRESHOLD = 0.5  # Lower means more strict matching

# States
STATE_SLIDESHOW = 0
STATE_PROCESSING = 1
STATE_RECOGNIZED = 2
STATE_UNRECOGNIZED = 3

current_state = STATE_SLIDESHOW
last_detection_time = 0
current_employee_id = None
last_slideshow_change = time.time()

# Function to get weather information
def get_weather_info():
    try:
        # Replace with your API key and location
        api_key = "27683695098cf44afc28e31152dffe5a"
        city = "Jinju"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            weather = {
                'temp': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity']
            }
            return weather
        else:
            return {'temp': 'N/A', 'description': 'Weather data unavailable', 'humidity': 'N/A'}
    except Exception as e:
        print(f"Weather API error: {e}")
        return {'temp': 'N/A', 'description': 'Weather data unavailable', 'humidity': 'N/A'}

# Function to speak text in background thread
def speak_text(text):
    try:
        #tts = gTTS(text=text, lang='en')
        tts = gTTS(text=text, lang='ko', slow=False)
        tts.save("welcome_audio.mp3")
        pygame.mixer.music.load("welcome_audio.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"TTS error: {e}")

# Audio worker thread
def audio_worker():
    while True:
        text = audio_queue.get()
        speak_text(text)
        audio_queue.task_done()

# Start audio worker thread
threading.Thread(target=audio_worker, daemon=True).start()

# Function to check for eye blinks (basic liveness detection)
def detect_blink(face_landmarks):
    try:
        # Get landmarks for left and right eye
        left_eye = face_landmarks[0]['left_eye']
        right_eye = face_landmarks[0]['right_eye']
        
        # Calculate the eye aspect ratio
        def eye_aspect_ratio(eye):
            # Compute the euclidean distances between the vertical eye landmarks
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            # Compute the euclidean distance between the horizontal eye landmarks
            C = np.linalg.norm(eye[0] - eye[3])
            # Compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)
            return ear
        
        # Average the eye aspect ratio together for both eyes
        ear_left = eye_aspect_ratio(np.array(left_eye))
        ear_right = eye_aspect_ratio(np.array(right_eye))
        ear = (ear_left + ear_right) / 2.0
        
        # Check if eyes are closed (lower EAR means more closed)
        if ear < 0.2:  # Threshold for "closed"
            return True
            
        return False
    except Exception as e:
        print(f"Blink detection error: {e}")
        return False  # Default to not detecting a blink

# Function to load employee image from local storage
def load_employee_image(employee_id):
    image_path = os.path.join(employee_images_dir, f"{employee_id}.jpg")
    if not os.path.exists(image_path):
        # Try png if jpg not found
        image_path = os.path.join(employee_images_dir, f"{employee_id}.png")
        if not os.path.exists(image_path):
            return None
    
    return cv2.imread(image_path)

# Main processing loop
while True:
    current_time = time.time()
    
    # STATE: SLIDESHOW - Show rotating images when no one is detected
    if current_state == STATE_SLIDESHOW:
        # Load and display the next slideshow image
        if current_time - last_slideshow_change > SLIDESHOW_INTERVAL:
            current_slideshow_index = (current_slideshow_index + 1) % len(slideshow_images)
            slideshow_img = cv2.imread(slideshow_images[current_slideshow_index])
            slideshow_img = cv2.resize(slideshow_img, (1080, 1920))
            last_slideshow_change = current_time
            
        # Show the current slideshow image
        cv2.imshow('Welcome System', slideshow_img)
    
    # Process camera feed regardless of state
    success, img = cap.read()
    if not success:
        print("Failed to get camera feed")
        continue
        
    # Resize for faster processing
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    # Detect faces in current frame
    face_current_frame = face_recognition.face_locations(img_small)
    
    # If faces are detected and we're in slideshow mode, switch to processing
    if face_current_frame and current_state == STATE_SLIDESHOW:
        current_state = STATE_PROCESSING
        print("Face detected, processing...")
    
    # If no faces detected for RECOGNITION_TIMEOUT seconds, return to slideshow
    if not face_current_frame and current_state != STATE_SLIDESHOW:
        if current_time - last_detection_time > RECOGNITION_TIMEOUT:
            current_state = STATE_SLIDESHOW
            current_employee_id = None
            print("No face detected for timeout period, returning to slideshow")
    else:
        last_detection_time = current_time
    
    # STATE: PROCESSING - Process detected faces
    if current_state == STATE_PROCESSING and face_current_frame:
        # Get face encodings
        encodings_current_frame = face_recognition.face_encodings(img_small, face_current_frame)
        face_landmarks = face_recognition.face_landmarks(img_small, face_current_frame)
        
        # Get full size image for display
        imgBackground = cv2.imread('resources/cover.jpg')
        imgBackground[162:162 + img.shape[0], 55:55 + img.shape[1]] = img
        
        # Process each detected face
        for encode_face, face_loc, landmarks in zip(encodings_current_frame, face_current_frame, face_landmarks):
            # Check for liveness (blink detection)
            is_live = 1 #detect_blink(face_landmarks)
            
            if not is_live:
                print("Liveness check failed - possible photo attack")
                # Draw red box to indicate possible spoofing
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(imgBackground, (55 + x1, 162 + y1), (55 + x2, 162 + y2), (0, 0, 255), 2)
                cv2.putText(imgBackground, "POSSIBLE FAKE", (55 + x1, 162 + y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue
                
            # Compare with known faces
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_distance = face_recognition.face_distance(encode_list_known, encode_face)
            
            # Find best match
            best_match_index = np.argmin(face_distance)
            
            # If we have a good match
            if matches[best_match_index] and face_distance[best_match_index] < FACE_CONFIDENCE_THRESHOLD:
                # Get employee ID and info
                employee_id = employee_ids[best_match_index]
                current_employee_id = employee_id
                
                # Get employee data from Firebase
                employee_info = db.reference(f'Employees/{employee_id}').get()
                
                if employee_info:
                    # Get employee image from local storage
                    employee_image = load_employee_image(employee_id)
                    
                    if employee_image is None:
                        print(f"Warning: Image for employee {employee_id} not found")
                        # Use a placeholder image or just continue without image
                        employee_image = np.zeros((216, 216, 3), dtype=np.uint8)
                    
                    # Update state to recognized
                    current_state = STATE_RECOGNIZED
                    
                    # Get weather info
                    weather = get_weather_info()
                    
                    # Get personalized messages
                    messages = employee_info.get('messages', [])
                    if messages:
                        latest_message = messages[0]
                    else:
                        latest_message = "No new messages"
                    
                    # Display employee info on the welcome screen
                    # Display name
                    cv2.putText(imgBackground, f"Welcome, {employee_info['name']}", 
                               (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 2)
                    
                    # Display weather
                    cv2.putText(imgBackground, f"Weather: {weather['temp']} C, {weather['description']}", 
                               (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
                    # Display message
                    cv2.putText(imgBackground, f"Message: {latest_message[:50]}{'...' if len(latest_message) > 50 else ''}", 
                               (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
                    # Display employee image
                    employee_image_resized = cv2.resize(employee_image, (200, 200))
                    imgBackground[250:250+200, 400:400+200] = employee_image_resized
                    
                    # Queue welcome message for audio playback
                    gender_prefix = "Mr." if employee_info.get('gender', 'M') == 'M' else "Ms."
                    welcome_text = f"{gender_prefix} {employee_info['name']}, welcome to ABC Company."
                    if not audio_queue.qsize():  # Only queue if not already speaking
                        audio_queue.put(welcome_text)
                    
                    # Update last visit timestamp in Firebase
                    db.reference(f'Employees/{employee_id}/last_visit').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
            else:
                # Not recognized - offer registration
                current_state = STATE_UNRECOGNIZED
                
                # Crop detected face for display
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                face_img = img[y1:y2, x1:x2]
                
                # Display on welcome screen
                cv2.putText(imgBackground, "Welcome to UNID!", 
                           (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(imgBackground, "Would you like to register?", 
                           (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Display detected face
                try:
                    face_img_resized = cv2.resize(face_img, (200, 200))
                    imgBackground[250:250+200, 400:400+200] = face_img_resized
                except:
                    pass
                
                # Display Yes/No buttons
                cv2.rectangle(imgBackground, (400, 500), (500, 550), (0, 255, 0), -1)
                cv2.putText(imgBackground, "Yes", (425, 535), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.rectangle(imgBackground, (550, 500), (650, 550), (0, 0, 255), -1)
                cv2.putText(imgBackground, "No", (585, 535), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Queue generic welcome message for audio playback
                # welcome_text = "Welcome to UNID. If you would like to be welcomed by name next time, please register."
                welcome_text = "오서 오십시오!"
                if not audio_queue.qsize():  # Only queue if not already speaking
                    audio_queue.put(welcome_text)
                
                # Here you would normally implement touch screen logic to capture the response
                # For this example, we'll just show the option
    
    # Display the appropriate screen based on state
    if current_state != STATE_SLIDESHOW:
        cv2.imshow('Welcome System', imgBackground)
    
    # Check for exit key
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()