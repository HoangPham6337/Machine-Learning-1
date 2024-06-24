import cv2
import uuid
import os

labels = ["thumbsup", "thumbsdown", "thankyou", "livelong", "okay"]
number_of_imgs = 400

IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")
SOURCE_IMAGES_PATH = os.path.join(".", "Test")

# Ensure the target directory structure exists
for label in labels:
    label_path = os.path.join(IMAGES_PATH, label)
    os.makedirs(label_path, exist_ok=True)

# Read images from files
for label in labels:
    print(f"Collecting images for {label}")
    
    label_source_path = os.path.join(SOURCE_IMAGES_PATH, label)
    image_files = [f for f in os.listdir(label_source_path) if os.path.isfile(os.path.join(label_source_path, f))]
    
    for img_idx, image_file in enumerate(image_files[:number_of_imgs]):
        print(f"Iteration {img_idx}: Collecting image from {image_file}")
        
        image_path = os.path.join(label_source_path, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read image {image_path}. Skipping.")
            continue
        
        img_name = os.path.join(IMAGES_PATH, label, f"{label}.{uuid.uuid1()}.jpg")
        
        # cv2.imwrite(img_name, frame)
        # cv2.imshow("Collected Image", frame)
        # cv2.waitKey(500)  # Display each image for 500ms

cv2.destroyAllWindows()