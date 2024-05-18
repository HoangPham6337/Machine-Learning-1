import os
labels = ["thumbsup", "thumbsdown", "thankyou", "livelong"]
number_of_imgs = 10

IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")

# Uncomment this to clean the image folders
# if os.path.exists(IMAGES_PATH):
#     os.system(f"rm -r {IMAGES_PATH}")

if not os.path.exists(IMAGES_PATH):
    os.system(f"mkdir -p {IMAGES_PATH}")

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.system(f"mkdir {path}")
