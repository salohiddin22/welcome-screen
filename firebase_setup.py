import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import datetime
import os

def setup_firebase_database():
    """Initialize Firebase and set up sample data structure"""
    # Initialize Firebase with your service account
    cred = credentials.Certificate("service_account_key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://welcome-screen-9366d-default-rtdb.firebaseio.com/',
        # 'storageBucket': 'welcome-screen-9366d.appspot.com'
    })
    
    # Reference to root of database
    ref = db.reference('/')
    
    # Sample employee data
    employees_data = {
        "Employees": {
            "emp001": {
                "name": "Donald Trump",
                "gender": "M",
                "position": "President",
                "department": "Management",
                "messages": ["Meeting at in Conference Room A"],
                "last_visit": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },

            "emp002": {
                "name": "Dean",
                "gender": "M",
                "position": "AI Researcher",
                "department": "서루션팀",
                "messages": ["Quarterly report due tomorrow"],
                "last_visit": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        },
        "SystemSettings": {
            "slideshow_interval": 5,
            "recognition_timeout": 10,
            "weather_location": "Jinju",
            "company_name": "UNID Company"
        }
    }
    
    # Set the data
    ref.set(employees_data)
    print("Firebase database initialized with sample data.")
    
    # Print employee IDs for reference
    employee_ids = employees_data["Employees"].keys()
    print(f"Created sample employees with IDs: {', '.join(employee_ids)}")
    print("Make sure your images match these IDs exactly (e.g., emp001.jpg)")

def setup_local_directories():
    """Set up local directories for images and resources"""
    # Create employee images directory
    employee_images_dir = "employee_images"
    if not os.path.exists(employee_images_dir):
        os.makedirs(employee_images_dir)
        print(f"Created directory: {employee_images_dir}")
        print("Please add employee images to this directory named as emp001.jpg, emp002.jpg, etc.")
    
    # Create slideshow images directory
    slideshow_dir = "slideshow_images"
    if not os.path.exists(slideshow_dir):
        os.makedirs(slideshow_dir)
        print(f"Created directory: {slideshow_dir}")
        print("Please add slideshow images to this directory")
    
    # Create resources directory
    resources_dir = "resources"
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
        print(f"Created directory: {resources_dir}")
    
    # Create blank welcome background if it doesn't exist
    background_file = os.path.join(resources_dir, "cover.jpg")
    if not os.path.exists(background_file):
        # Create a blank background (1080x1920) - you'll want to replace this with a proper design
        import numpy as np
        import cv2
        background = np.ones((1920, 1080, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(background, "Welcome to ABC Company", (200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.imwrite(background_file, background)
        print(f"Created placeholder welcome background: {background_file}")
        print("You should replace this with your custom designed background")

def create_sample_images():
    """Create placeholder images for testing if needed"""
    import numpy as np
    import cv2
    
    employee_images_dir = "employee_images"
    
    # Only create samples if directory is empty
    if len([f for f in os.listdir(employee_images_dir) if f.endswith(('.jpg', '.png'))]) == 0:
        print("Creating placeholder employee images for testing...")
        
        # Create simple colored images with labels for each sample employee
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
        
        for i, emp_id in enumerate(['emp001', 'emp002', 'emp003']):
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
            
            # Add colored rectangle
            cv2.rectangle(img, (50, 50), (350, 350), colors[i % len(colors)], -1)
            
            # Add text
            cv2.putText(img, f"Sample {emp_id}", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save image
            cv2.imwrite(os.path.join(employee_images_dir, f"{emp_id}.jpg"), img)
        
        print("Created placeholder images. Replace these with actual employee photos.")
    
    # Create sample slideshow images if needed
    slideshow_dir = "slideshow_images"
    if len([f for f in os.listdir(slideshow_dir) if f.endswith(('.jpg', '.png'))]) == 0:
        print("Creating sample slideshow images...")
        
        # Create 3 sample slideshow images
        for i in range(1, 4):
            img = np.ones((1920, 1080, 3), dtype=np.uint8) * 240  # Light gray
            
            # Add text
            cv2.putText(img, f"Slideshow Image {i}", (400, 960), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            # Save image
            cv2.imwrite(os.path.join(slideshow_dir, f"slide{i}.jpg"), img)
        
        print("Created sample slideshow images. Replace these with your desired slideshow content.")

if __name__ == "__main__":
    print("Setting up Face Recognition Welcome System...")
    setup_firebase_database()
    setup_local_directories()
    create_sample_images()
    print("\nSetup complete! Next steps:")
    print("1. Add real employee photos to the 'employee_images' directory")
    print("2. Run the face_encoder.py script to encode employee faces")
    print("3. Run the welcome_system.py script to start the welcome system")