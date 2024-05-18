import cv2
import uuid
import os
import time

labels = ["thumbsup", "thumbsdown", "thankyou", "livelong"]
number_of_imgs = 10

IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")

# Capture images
capture = cv2.VideoCapture(0)
input("Press Enter to begin")
for label in labels:
    print(f"Collecting images for {label}")
    
    for img in range(number_of_imgs):
        print(f"Iteration {img}: Collecting image")
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture image. Aborting!")
            exit(1)
        
        img_name = os.path.join(IMAGES_PATH, label,f"{label}.{uuid.uuid1()}.jpg")

        cv2.imwrite(img_name, frame)
        cv2.imshow("Captured Image", frame)
        cv2.waitKey()

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()