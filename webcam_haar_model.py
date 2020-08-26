"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import copy
import matplotlib.pyplot as plt
import torch
from models import Net


def show_webcam(model, face_cascade):
    cam = cv2.VideoCapture(0)
    face = None
    while True:
        ret_val, img = cam.read()

        faces = face_cascade.detectMultiScale(img, 1.2, 2, minSize=(100, 100))
        image_with_detections = img.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_detections, (x, y),
                          (x+w, y+h), (255, 0, 0), 3)
            break

        for i, (x, y, w, h) in enumerate(faces):

            # Set padding to approximately 20% of total width
            p = w//5

            # Select the region of interest that is the face in the image
            if (y-p) >= 0 and (y+h+p) <= img.shape[0] and (x-p) >= 0 and (x+w+p) <= img.shape[1]:
                p = 0
            roi = img[y-p:y+h+p, x-p:x+w+p]

            # TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)

            scale = [roi.shape[1]/224, roi.shape[0]/224]

            if roi.shape[0] < 10 or roi.shape[1] < 10:
                break

            roi = cv2.resize(roi, (224, 224))
            face = copy.copy(roi)

            # TODO: Convert the face region from RGB to grayscale
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            # TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
            roi = roi/255

            # TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
            roi_tensor = torch.tensor(roi).view(1, 1, 224, 224).float()

            # TODO: Make facial keypoint predictions using your loaded, trained network
            keypoints_normalized = model(
                roi_tensor).view(68, 2).detach().numpy()

            keypoints = (keypoints_normalized * 50 + 100) * scale

            for kp in keypoints:
                xy = int(kp[0]) + x + p, int(kp[1]) + y + p
                cv2.circle(image_with_detections, xy, 3, (0, 255, 0), -1)

            break

        cv2.imshow('result', image_with_detections)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier(
        'detector_architectures/haarcascade_frontalface_default.xml')

    model = Net()
    model.load_state_dict(torch.load('./saved_models/keypoints_model_1.pt'))
    model.eval()

    show_webcam(model, face_cascade)


if __name__ == '__main__':
    main()
