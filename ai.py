from ultralytics import YOLO

import numpy as np

def recognize(img):
    model = YOLO("last.pt")
    results = model(img)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    return(names_dict[np.argmax(probs)])