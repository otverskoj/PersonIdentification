import os
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

    def embed(self, person_bboxes: np.ndarray) -> np.ndarray:
        return np.array([])
