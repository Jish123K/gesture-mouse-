import torch

import torchvision.transforms as transforms

import cv2

import time

import math

# Load the model

model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)

model.eval()

# Define the transformation pipeline for the input images

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

])

class handDetector():

    def __init__(self, detectionCon=0.5):

        self.detectionCon = detectionCon

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():

            output = model(img_tensor)['out'][0]

        output = output.argmax(0).byte().cpu().numpy()

        output = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        handContours = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 500:

                continue

            handContours.append(contour)

        if len(handContours) == 0:

            return img

        if draw:

            cv2.drawContours(img, handContours, -1, (0, 255, 0), 3)

        return img

    def findPosition(self, img, handNo=0, draw=True):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:

        return [], []

    handContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    hull = cv2.convexHull(handContour, returnPoints=False)

    defects = cv2.convexityDefects(handContour, hull)

    fingers = []

    for i in range(defects.shape[0]):

        s, e, f, d = defects[i, 0]

        start = tuple(handContour[s][0])

        end = tuple(handContour[e][0])

        far = tuple(handContour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)

        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        if angle <= math.pi / 2:

            fingers.append(far)

    fingers = sorted(fingers, key=lambda x: x[1])

    if draw:

        for fingerPos in fingers:

            cv2.circle(img, fingerPos, 10, (0, 255, 0), 2)

    return img, fingers

        

            

                

