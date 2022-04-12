import cv2
import numpy as np
from openvino.runtime import Core
import matplotlib.pyplot as plt

class DepthEstimator:
    def __init__(self) -> None:
        self._init_model()

    def _init_model(self) -> None:
        # self.topology_file = r'model_data\midasnet\converted\public\midasnet\FP32\midasnet.xml'
        # self.weights_file = r'model_data\midasnet\converted\public\midasnet\FP32\midasnet.bin'
        #
        # ie = Core()
        # model = ie.read_model(self.topology_file, weights=self.weights_file)
        #
        # self.model = ie.compile_model(model, device_name='CPU')
        #
        # self.input_layer = next(iter(self.model.inputs))
        # self.output_layer = next(iter(self.model.outputs))
        # self.n_batch, self.n_channels, self.net_h, self.net_w = self.input_layer.shape
        self.model = cv2.dnn.readNetFromONNX("model_data/midasnet/converted/public/midasnet/midasnet.onnx")

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.transpose(cv2.resize(frame.astype(np.float32), (384, 384)), (2, 1, 0)), 0)

    def process_frame(self, frame) -> np.ndarray:
        input_data = self._prepare_frame(frame)
        return self.model([input_data])[self.output_layer][0]


if __name__ == '__main__':

    model = cv2.dnn.readNetFromONNX("model_data/midasnet/converted/public/midasnet/midasnet.onnx")
    input_img = cv2.imread("data/input/image/frame.jpg", cv2.IMREAD_COLOR)
    orig_h, orig_w = input_img.shape[:2]
    input_img = input_img.astype(np.float32)
    input_img = cv2.resize(input_img, (384, 384))

    # define preprocess parameters
    mean = np.array([123.675, 116.28, 103.53])
    std = [51.525, 50.4, 50.625]

    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        size=(384, 384),  # img target size
        mean=mean,
        swapRB=False,  # BGR -> RGB
    )
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    model.setInput(input_blob)
    # OpenCV DNN inference
    out = model.forward()
    out = np.transpose(out, (1, 2, 0))
    out = out - np.min(out)
    out /= np.max(out)
    out = 1 - out

    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)
    out = cv2.resize(out, (orig_w, orig_h))
    plt.imshow(out)
    plt.show()
