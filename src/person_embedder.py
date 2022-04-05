import os
from turtle import resizemode
from urllib import request
import cv2
import numpy as np
from openvino.runtime import Core


class PersonEmbedder:
    def __init__(self, model_data_path: str = None) -> None:
        self.model_data_path = model_data_path or \
            r'model_data\person-reidentification-retail-0286\intel\person-reidentification-retail-0286\FP16'
        self.init_model()
    
    def init_model(self) -> None:
        filenames = list(os.walk(self.model_data_path))[0][2]
        for filename in filenames:
            tmp = os.path.join(self.model_data_path, filename)
            if filename.endswith('.bin'):
                weights_file = tmp
            elif filename.endswith('.xml'):
                topology_file = tmp

        ie = Core()
        model = ie.read_model(topology_file, weights=weights_file)
        self.model = ie.compile_model(model, device_name='CPU')

        self.input_layer = next(iter(self.model.inputs))
        self.output_layer = next(iter(self.model.outputs))
        self.n_batch, self.n_channels, self.net_h, self.net_w = self.input_layer.shape
        
    def embed(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, (self.net_w, self.net_h))
        input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        return self.model([input_data])[self.output_layer]
    
    def embed_persons(self, image: np.ndarray, person_bboxes: np.ndarray) -> np.ndarray:
        result = []
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            person = image[y1:y2 + 1, x1:x2 + 1]
            embedding = self.embed(person)
            result.append(embedding)
        return np.array(result)


if __name__ == '__main__':
    pe = PersonEmbedder()
    result = pe.embed(cv2.imread(r'D:\Work\PersonIdentification\data\input\image\frame.jpg'))
    print(result.shape)
