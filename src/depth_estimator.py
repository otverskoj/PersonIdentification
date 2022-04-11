import cv2
import numpy as np
from openvino.runtime import Core
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type
from openvino.preprocess import ResizeAlgorithm


class DepthEstimator:
    def __init__(self) -> None:
        self._init_model()
    
    def _init_model(self) -> None:
        self.topology_file = r'model_data\midasnet\converted\public\midasnet\FP32\midasnet.xml'
        self.weights_file = r'model_data\midasnet\converted\public\midasnet\FP32\midasnet.bin'
        
        ie = Core()
        model = ie.read_model(self.topology_file, weights=self.weights_file)
        
        # ppp = PrePostProcessor(model)
        # ppp.input().tensor() \
        #     .set_element_type(Type.u8) \
        #     .set_shape([1, 1440, 2560, 3]) \
        #     .set_layout(Layout('NHWC')) \
        #     .set_color_format(ColorFormat.BGR)
        # ppp.input().model().set_layout(Layout('NCHW'))
        # ppp.input().preprocess() \
        #     .convert_element_type(Type.f32) \
        #     .resize(ResizeAlgorithm.RESIZE_LINEAR)
        # # print(f'Dump preprocessor: {ppp}')
        # model = ppp.build()

        self.model = ie.compile_model(model, device_name='CPU')

        self.input_layer = next(iter(self.model.inputs))
        self.output_layer = next(iter(self.model.outputs))
        self.n_batch, self.n_channels, self.net_h, self.net_w = self.input_layer.shape
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.transpose(cv2.resize(frame.astype(np.float32), (384, 384)), (2, 1, 0)), 0)
    
    def process_frame(self, frame) -> np.ndarray:
        input_data = self._prepare_frame(frame)
        return self.model([input_data])[self.output_layer][0]


if __name__ == '__main__':
    de = DepthEstimator()
    depth_map = de.process_frame(cv2.imread(r'data\input\image\frame.jpg'))
    depth_map = cv2.resize(depth_map, (800, 600))
    depth_map = depth_map.astype(np.uint8)
    cv2.imshow('res', cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO))
    cv2.waitKey()
