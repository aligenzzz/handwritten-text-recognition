import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from constants import LABELS_PATH, WEIGHTS_PATH, CONFIG_PATH

net = None
ln = None


def load_yolo_model():
    global net, ln
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    return ln


def get_yolo_boxes(image, confidence=0.2, threshold=0.3):
    global net, ln

    if ln is None and net is None:
        load_yolo_model()

    # for boxes' output ...
    # labels = open(LABELS_PATH).read().strip().split('\n')
    # initialize a list of colors to represent each possible class label
    # np.random.seed(42)
    # colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # image = cv2.imread(image)
    (H, W) = image.shape[:2]

    # Blob (Binary Large Object)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_ = scores[class_id]

            if confidence_ > confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence_))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    characters = []
    # copied_image = image

    if len(idxs) > 0:
        for i in idxs:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # ... for boxes' output
            # color = [int(c) for c in colors[class_ids[i]]]
            # cv2.rectangle(copied_image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            # cv2.putText(copied_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)

            characters.append([x, x + w, y, y + h])

    # cv2.imshow(copied_image)
    # cv2.waitKey(0)

    return image, characters
