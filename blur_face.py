# USAGE
# python blur_face.py --folder images --face face_detector --method simple
# python blur_face.py --folder images --face face_detector --method pixelated

import argparse
import os

import cv2
import numpy as np

# import the necessary packages
from utils.face_blurring import anonymize_face_pixelate, anonymize_face_simple

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--folder", required=True, help="path to input images folder")
ap.add_argument(
    "-f", "--face", required=True, help="path to face detector model directory"
)
ap.add_argument(
    "-m",
    "--method",
    type=str,
    default="simple",
    choices=["simple", "pixelated"],
    help="face blurring/anonymizing method",
)
ap.add_argument(
    "-b",
    "--blocks",
    type=int,
    default=20,
    help="# of blocks for the pixelated blurring method",
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak detections",
)
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# create output directory if it doesn't exist
output_dir = os.path.join(args["folder"], "blurred")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all images from the input folder
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
for filename in os.listdir(args["folder"]):
    if not filename.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(args["folder"], filename)
    print(f"[INFO] processing {filename}...")

    # load the input image from disk, clone it, and grab the image spatial dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]

            # apply the appropriate face blurring method
            if args["method"] == "simple":
                face = anonymize_face_simple(face, factor=3.0)
            else:
                face = anonymize_face_pixelate(face, blocks=args["blocks"])

            # store the blurred face in the output image
            image[startY:endY, startX:endX] = face

    # save only the blurred image
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

print("[INFO] Processing complete! Blurred images saved in the 'blurred' subdirectory")
