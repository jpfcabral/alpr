import cv2
import utils
import time

WIDTH_DEFAULT = 416
HEIGHT_DEFAULT = 416
CONFIDENCE_DEFAULT = 0.5
NMS_DEFAULT = 0.5


class LPRNet:
    def __init__(self, config_path, weights_path, names_path, cuda_en=False):

        self.__config_path = config_path
        self.__weights_path = weights_path
        self.__names_path = names_path

        self.__net_width = WIDTH_DEFAULT
        self.__net_height = HEIGHT_DEFAULT
        self.__net_width, self.__net_height = self.__get_dimensions()
        self.__confidence_threshold = CONFIDENCE_DEFAULT
        self.__nms_threshold = NMS_DEFAULT
        self.__names = self.__get_names()

        self.__net = cv2.dnn.readNetFromDarknet(self.__config_path, self.__weights_path)
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) if cuda_en else self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) if cuda_en else self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.__lp_net_layer_names = self.__net.getLayerNames()
        self.__lp_net_disconnected_layers = [self.__lp_net_layer_names[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

        self.__image = []
        self.__net_detections = []  # Detecções encontradas
        self.__cropped_detections = []  # Imagens cortadas
        self.__net_time = None  # Tempo de inferência

    def detect(self, **kwargs):
        if self.__image == []:
            return ''
        start_time = time.perf_counter()
        blob = cv2.dnn.blobFromImage(self.__image, 1 / 255, (self.__net_width, self.__net_height), [0, 0, 0], 1, crop=False)
        self.__net.setInput(blob)
        lp_net_out = self.__net.forward(self.__lp_net_disconnected_layers)
        self.__net_detections = utils.find_objects(lp_net_out, self.__confidence_threshold)
        self.__net_detections = utils.normalized_to_absolut_coordinates(self.__net_detections, self.__image.shape)
        self.__net_detections = utils.nms(self.__net_detections, self.__confidence_threshold, self.__nms_threshold)
        stop_time = time.perf_counter()
        self.__net_time = round(stop_time - start_time, 3)
        return self.__net_detections

    def set_confidence_threshold(self, c):
        self.__confidence_threshold = c if 0 <= c <= 1 else CONFIDENCE_DEFAULT

    def set_nms_threshold(self, nms):
        self.__nms_threshold = nms if 0 <= nms <= 1 else NMS_DEFAULT

    def set_input(self, image_path):
        aux = cv2.imread(image_path)
        self.__image = aux if utils.sizetest(aux) else []
    
    def set_image(self, image):
        self.__image = image if utils.sizetest(image) else []

    def set_dimensions(self, width, height):
        self.__net_width = width
        self.__net_height = height

    def get_plate(self, plate_type):
        if len(self.__net_detections) != 7:
            return ''
        else:
            return utils.get_plate_char(self.__net_detections, self.__names, plate_type)

    def __get_dimensions(self):
        """
        Entra na página de configuração da rede e pega os valores de width e height
        :return: widht, height from net
        """
        with open(self.__config_path) as file:
            for line in file:
                if line.split('=')[0] == 'height':
                    height = int(line.split('=')[1])
                if line.split('=')[0] == 'width':
                    width = int(line.split('=')[1])
        return width, height

    def __get_names(self):
        with open(self.__names_path) as file:
            names = [line.strip() for line in file.readlines()]
        return names

    def get_cropped_images_from_detections(self):
        return utils.cortar(self.__image, self.__net_detections)

    def get_net_detections(self):
        return self.__net_detections

    def get_inference_time(self):
        return self.__net_time
