import os
from typing import Tuple
import numpy as np
import cv2


class PersonDetector:
    def __init__(self,
                 model_data_path: str = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45) -> None:
        self.model_data_path = model_data_path or r'model_data\yolov5'
        self.confidence_threshold = confidence_threshold
        self.nms_thershold = nms_threshold
        self.init_model()

    def init_model(self):
        filenames = list(os.walk(self.model_data_path))[0][2]
        for filename in filenames:
            tmp = os.path.join(self.model_data_path, filename)
            if filename.endswith('.txt'):
                classes_file = tmp
            elif filename.endswith('.onnx'):
                onnx_file = tmp

        with open(classes_file, mode='r', encoding='utf-8') as f:
            self.classes = tuple(c.strip() for c in f.readlines())
        self.model = cv2.dnn.readNetFromONNX(onnx_file)

    def detect_persons(self, image: np.ndarray = None) -> Tuple:    
        yolov5_dim = 640

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (yolov5_dim, yolov5_dim)), 
                                     1 / 255.0, (yolov5_dim, yolov5_dim))        
        self.model.setInput(blob)
        predictions = self.model.forward()
        output = predictions[0]

        rows = output.shape[0]

        x_factor = width / yolov5_dim
        y_factor =  height / yolov5_dim

        bboxes, confidences, class_ids = [], [], []
        for row_idx in range(rows):
            row = output[row_idx]
            confidence = row[4]
            if confidence > self.confidence_threshold:
                classes_scores = row[5:]
                _, _, _, max_idx = cv2.minMaxLoc(classes_scores)
                class_id = max_idx[1]
                if classes_scores[class_id] > 0.25 and self.classes[class_id] == 'person':
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    bbox = np.array([left, top, width, height])
                    
                    bboxes.append(bbox)
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, 
                                   self.confidence_threshold, 
                                   self.nms_thershold)

        return np.array(
            [bboxes[i] for i in indices]
        )
