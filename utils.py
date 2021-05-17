import numpy as np
import cv2


def find_objects(layer_list, confidence_threshold):
    """
    Essa função irá escolher os bounding boxes que passarem no limite de confiança, bem como reoganizar os dados para
    que fiquem mais

    :param layer_list: Lista com layers de saida da rede [layer1, layer2, ... , layerN]
    :param confidence_threshold: Nivel de confiaça para detecção
    :return: Lista com detecções [detection][class, confidence, x, y, w, h]
    """
    detected_objects = []
    for layer in layer_list:
        for bounding_boxes_values in layer:

            # Guarda os valores de confidence para cada classe armazenados depois do quinto valor da lista
            detection_scores = bounding_boxes_values[5:]

            # Retorna a posição do maior valor numa lista de numeros (conficence)
            class_id = np.argmax(detection_scores)

            # Armazena o valor numerico do confidence
            confidence = round(detection_scores[class_id], 2)

            # Verifica se a classe com maior confiança é maior que nosso limite de confiança
            if confidence > confidence_threshold:
                x = bounding_boxes_values[0]
                y = bounding_boxes_values[1]
                w = bounding_boxes_values[2]
                h = bounding_boxes_values[3]
                detected_objects.append([class_id, confidence, x, y, w, h])

    return detected_objects


def normalized_to_absolut_coordinates(detection_list, img_shape):
    """
    Transforma coordenadas normalizadas para coordenadas absolutas

    :param detection_list: Lista com coordenadas normalizadas [class, confidence, x, y, w, h], [class, confidence,...
    :param img_shape: Lista com tamanho da imagem [widht, height, channels]
    :return: Lista com coordenadas absolutas [class, confidence, x, y, w, h], [class, confidence, x, y, w, h], ...
    """
    absolut_bbox = []
    img_height, img_widht, img_channels = img_shape
    for detection in detection_list:
        w = int(detection[4] * img_widht)
        h = int(detection[5] * img_height)
        x = int((detection[2] * img_widht) - (w/2))
        y = int((detection[3] * img_height) - (h/2))
        absolut_bbox.append([detection[0], detection[1], x, y, w, h])

    return absolut_bbox


def absolut_to_normalized_coordinates(detections_list, img_shape):
    """
    Transforma coordenadas absolutas para coordenadas normalizadas

    :param detections_list: Lista com coordenadas absolutas [class, confidence, x, y, w, h]
    :param img_shape: Lista com tamanho da imagem [widht, height, channels]
    :return: Lista com coordenadas normalizadas [class, confidence, x, y, w, h]
    """
    # print(type(detection[4]))
    normalized_bbox = []
    img_height, img_widht, img_channels = img_shape
    for detection in detections_list:
        w = detection[4] / img_widht
        h = detection[5] / img_height
        x = (detection[2] / img_widht) + (w/2)
        y = (detection[3] / img_height) + (h/2)
        normalized_bbox.append([detection[0], detection[1], x, y, w, h])

    return normalized_bbox


def draw_detections(image, detection_list, classes):
    """
    Desenha os bbs e classes na imagem

    :param image: Imagem
    :param detection_list: Lista de detecções em VALORES ABSOLUTOS
    :param classes: Classes
    :return:
    """
    placa = ''
    nome = ""
    # detection_list = sort_detections_by_x_value(detection_list)
    for detection in detection_list:
        if detection[2] <= 0 or detection[3] <= 0 or detection[4] <= 0 or detection[5] <= 0:
            break
        # print(detection)
        nome += classes[detection[0]]
        x = detection[2]
        y = detection[3]
        w = detection[4]
        h = detection[5]
        # placa = detection[6]
        espessura = int(w * 0.02)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), espessura)
        # cv2.rectangle(image, (x, y), (x + w, y - h), (0, 0, 0), -1)
        # cv2.putText(image, placa, (x, y - int(h * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, w * 0.006,
        #             (0, 255, 0), thickness=int(w*0.02))

    cv2.imshow(nome, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_detections_final(image, plate_detection_list, plate_classes, plates):
    """
    Desenha bb da rede de placas e as classes encontradas da rede de caracteres
    :param image: Imagem
    :param plate_detection_list: Lista com bbs das placas encontradas
    :param plate_classes: classes da rede de placa
    :param plates: Lista com os caracteres das placas encontradas
    :return:
    """
    i = 0
    for detection in plate_detection_list:
        x = detection[2]
        y = detection[3]
        w = detection[4]
        h = detection[5]
        espessura = int(w * 0.02)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), espessura)
        cv2.rectangle(image, (x, y), (x + w, y - h), (0, 0, 0), -1)
        cv2.putText(image, plates, (x, y - int(h * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, w * 0.006,
                    (0, 255, 0), thickness=int(w*0.02))
        i += 1
    # cv2.imshow('final', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def nms(detection_list, confidence_threshold, nms_threshold):
    """
    Aplica o Non Maximun Supression num conjunto de bounding boxes

    :param detection_list: Lista com detecções em VALORES ABSOLUTOS
    :param confidence_threshold: Limiar de confiança
    :param nms_threshold: Limiar do NMS
    :return: Lista atualizada pós NMS
    """
    bbox = []
    confidences = []
    class_ids = []
    out = []

    # Desmembra
    for detection in detection_list:
        bbox.append(detection[2:])
        confidences.append(float(detection[1]))
        class_ids.append(detection[0])

    indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_threshold, nms_threshold)

    # Seleciona os bbs que passaram pelo nms
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        class_id = class_ids[i]
        conf = confidences[i]
        out.append([class_id, round(conf, 2), x, y, w, h])

    return out


def get_data_from_name(image_name):
    """
    Desmebra o nome da imagem nos dados disponiveis neles
    :param image_name: Nome da imagem
    :return: Lista com dados separados
    """
    nome = image_name.split(".")[0]
    nome_recebido = list(nome)
    ano = ''.join(nome_recebido[:4])
    mes = ''.join(nome_recebido[4:6])
    dia = ''.join(nome_recebido[6:8])
    hora = ''.join(nome_recebido[8:10])
    minuto = ''.join(nome_recebido[10:12])
    segundo = ''.join(nome_recebido[12:14])
    codigo = ''.join(nome_recebido[14:24])
    certeza = ''.join(nome_recebido[24:27])
    placa = ''.join(nome_recebido[27:34])
    posicao = ''.join(nome_recebido[34])
    classificao = ''.join(nome_recebido[35:37])
    velocidade = ''.join(nome_recebido[37:40])
    comprimento = ''.join(nome_recebido[40:43])
    sequencial = ''.join(nome_recebido[43:])

    return [ano, mes, dia, hora, minuto, segundo, codigo, certeza, placa, posicao, classificao, velocidade, comprimento,
            sequencial]


def sort_detections_by_x_value(detection_list):
    """
    Ordena as detecções pelo valor de X
    :param detection_list: Lista de detecções
    :return: Lista de detecções ordenadas
    """
    array = np.array(detection_list)
    # print(array)
    sorted_detections_by_x_value = array[np.argsort(array[:, 2])]
    # print(sorted_detections_by_x_value)
    sorted_detections_by_x_value = sorted_detections_by_x_value.tolist()

    for detection in sorted_detections_by_x_value:
        detection[0] = int(detection[0])
        detection[2] = int(detection[2])
        detection[3] = int(detection[3])
        detection[4] = int(detection[4])
        detection[5] = int(detection[5])

    return sorted_detections_by_x_value


def generate_txt_label(name, detection_list, path):
    """
    Gera arquivo txt para treinamento (padrão darknet/yolo)
    :param name: Nome da imagem
    :param detection_list:
    :param path:
    :return:
    """
    name = name.split(".")[0]
    final_name = path + '/obj/' + name + '.txt'

    for detection in detection_list:
        class_id = round(detection[0])
        x = round(detection[2], 5)
        y = round(detection[3], 5)
        w = round(detection[4], 5)
        h = round(detection[5], 5)

        with open(final_name, 'a') as file:
            file.write(str(class_id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")


def generate_train_txt(name, path):
    """
    Gera arquivo para treinamento padrao yolo
    :param name: nome da imagem
    :param path: diretorio onde salvar o txt
    :return:
    """
    with open(path + '/test.txt', 'a') as file:
        file.write('/content/YOLO_metric/data/obj/' + name + '\n')


def generate_test_txt(name, path):
    """
    Gera arquivo para treinamento padrao yolo
    :param name: nome da imagem
    :param path: diretorio onde salvar o txt
    :return:
    """
    with open(path + '/test.txt', 'a') as file:
        file.write('data/test/' + name + '\n')


def compara_resultados_placa(nome, detection_list, classes, plate_type):
    """
    Compara o resultado encontrado com o dado no nome da imagem
    :param nome: Nome da imagem
    :param detection_list: Lista de detecções [class, confidence, x, y, w, h]
    :param classes: Nome das classes, e.g: "["flor", "abajour"]"
    :return: 1 se é igual, se não 0
    """
    nome_encontrado = []
    nome_verdadeiro = get_data_from_name(nome)[8]
    nome_verdadeiro = list(nome_verdadeiro)

    # Ordena classes pelo valor de x
    detection_list = sort_detections_by_x_value(detection_list)

    for classe_id in detection_list:
        nome_encontrado.append(classes[classe_id[0]])

    # Ajusta os caracteres que se parecem
    # nome_encontrado = ajusta_char_placa(detection_list, classes, plate_type)
    # print(f'Nome encontrado {nome_encontrado}')
    # print(f'Nome verdadeiro {nome_verdadeiro}')
    return 1 if nome_encontrado == nome_verdadeiro else 0


def ajusta_char_placa(detection_list, classes_list, plate_type):
    """

    :param detection_list:
    :param classes_list:
    :return:
    """
    nome_encontrado = []
    classes_ajustada = []
    # print(classes_list)

    # Ordena classes pelo valor de x
    detection_list = sort_detections_by_x_value(detection_list)

    # Armazena apenas as classes
    for classe_id in detection_list:
        nome_encontrado.append(classes_list[classe_id[0]])

    # print(len(nome_encontrado))
    # Ajusta os caracteres que se parecem
    nome_encontrado = ajusta_letras(nome_encontrado)
    nome_encontrado = ajusta_numeros(nome_encontrado)

    # Caso a placa seja do modelo novo, é ajustado o 4º digito
    if not plate_type:
        nome_encontrado = ajusta_np(nome_encontrado)

    nome_encontrado = list(nome_encontrado)
    # print(nome_encontrado)

    # Armazena endereço das classes
    for letra in nome_encontrado:
        classes_ajustada.append(classes_list.index(letra))

    nome_encontrado = []

    for classe in classes_ajustada:
        nome_encontrado.append(classes_list[classe])

    return nome_encontrado


def ajusta_np(letras):
    if letras[4] == '0':
        letras[4] = 'O'
    elif letras[4] == '1':
        letras[4] = 'I'
    elif letras[4] == '5':
        letras[4] = 'S'
    elif letras[4] == '4':
        letras[4] = 'A'
    elif letras[4] == '6':
        letras[4] = 'S'
    print("Uma placa nova foi encontrada e houve correção no 4º digito")
    return letras


def ajusta_letras(letras):
  # Numeros que se parecem com algumas letras
  for i in range(0, 3):
    if letras[i] == '0':
      letras[i] = 'O'
    elif letras[i] == '1':
      letras[i] = 'I'
    elif letras[i] == '5':
      letras[i] = 'S'
    elif letras[i] == '4':
      letras[i] = 'A'
    elif letras[i] == '6':
      letras[i] = 'S'
    elif letras[i] == '8':
      letras[i] = 'B'

  return letras


def ajusta_numeros(numeros):
  for i in range(3, 7):
    if numeros[i] == 'O':
      numeros[i] = '0'
    elif numeros[i] == 'I':
      numeros[i] = '1'
    elif numeros[i] == 'Z':
      numeros[i] = '2'
    elif numeros[i] == 'D':
      numeros[i] = '0'
    elif numeros[i] == 'B':
      numeros[i] = '8'
    elif numeros[i] == 'X':
      numeros[i] = '8'
    elif numeros[i] == 'Q':
      numeros[i] = '0'
  return numeros


def cortar(image, detections_list):
    """
    Corta as detecções encontradas nas imagens e salva numa lista de imagens
    :param image: imagem completa
    :param detections_list: lista de detecções
    :return: lista com imagens cortadas
    """
    plate_list = []
    for detection in detections_list:
        x = detection[2] if detection[2] > 0 else 0
        y = detection[3] if detection[3] > 0 else 0
        w = detection[4] if detection[4] > 0 else 0
        h = detection[5] if detection[5] > 0 else 0

        plate_list.append(image[y:y+h, x:x+w])

    return plate_list


def get_plate_char(char_detections, classes, plate_type):
    plate = ''
    char_detections = sort_detections_by_x_value(char_detections)
    char_detections = ajusta_char_placa(char_detections, classes, plate_type)

    char_detections = list(char_detections)

    # Verifica se os 3 primeiros digitos sao letras e os 4 ultimos sao numeros
    # if plate_type:
    #     for i in range(0, 3):
    #         if char_detections[i].isdigit():
    #             return ''
    #
    #     for i in range(3, 7):
    #         if not char_detections[i].isdigit():
    #             return ''
    # else:
    #     for i in range(0, 3):
    #         if char_detections[i].isdigit():
    #             return ''
    #     if not char_detections[3].isdigit():
    #         return ''
    #     if char_detections[4].isdigit():
    #         return ''
    #     for i in range(5, 7):
    #         if not char_detections[i].isdigit():
    #             return ''

    for char in char_detections:
        plate += char

    return plate


def draw_detected_plates(image, plates):
    box_size = 30 * len(plates)
    cv2.rectangle(image, (0, 0), (150, box_size), (0, 0, 0), -1)
    posicao = 0
    for plate in plates:
        cv2.putText(image, plate, (0, 25+(posicao*10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=3)
        posicao += 3


def sizetest(img):
    if type(img).__module__ == np.__name__:
        y, x, c = img.shape
        if y <= 0:
            return 0
        elif x <= 0:
            return 0
        else:
            return 1
    else:
        return 0
