#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import os

# Check https://www.fypsolutions.com/opencv-python/ssdlite-mobilenet-object-detection-with-opencv-dnn/

COCO_labels = { 0: 'background',
    1: '"person"', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant',12: 'street sign', 13: 'stop sign', 14: 'parking meter',
    15: 'zebra', 16: 'bird', 17: 'cat', 18: 'dog',19: 'horse',20: 'sheep',21: 'cow',22: 'elephant',
    23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella',29: 'shoe',
    30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror',
    67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 91: 'hair brush'}


def load_mobilenet():
    """Não mude ou renomeie esta função
        Carrega o modelo e os parametros da MobileNet. 
        Retorna a rede carregada.
    """
    proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt"
    model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(proto, model)

    return net


def detect(net, frame, CONFIDENCE, COLORS, CLASSES):
    """
        Recebe:
            net - a rede carregada
            frame - uma imagem colorida BGR
            CONFIDENCE - o grau de confiabilidade mínima da detecção
            COLORS - as cores atribídas a cada classe
            CLASSES - o array de classes
        Devolve: 
            img - a imagem com os objetos encontrados
            resultados - os resultados da detecção no formato
             [(label, score, point0, point1),...]
    """
    img = frame.copy()
    resultados = []
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # print("[INFO] {}".format(label))
                cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                resultados.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY)))

    return img, resultados


def separar_caixa_entre_animais(img, resultados):
    """Não mude ou renomeie esta função
        recebe o resultados da MobileNet e retorna dicionario com duas chaves, 'vaca' e 'lobo'.
        Na chave 'vaca' tem uma lista de cada caixa que existe uma vaca, no formato: [ [min_X, min_Y, max_X, max_Y] , [min_X, min_Y, max_X, max_Y] , ...]. Desenhe um retângulo azul em volta de cada vaca
        Na chave 'lobo' tem uma lista de uma unica caixa que engloba todos os lobos da imagem, no formato: [min_X, min_Y, max_X, max_Y]. Desenhe um retângulo vermelho em volta dos lobos

    """
    img = img.copy()

    animais = {}
    animais['vaca'] = []
    animais['lobo'] = []

    lobos = {}
    lobos['min_x'] = []
    lobos['min_y'] = []
    lobos['max_x'] = []
    lobos['max_y'] = []
    

    for detection in resultados:
        id, CONFIDENCE, start, end = detection

        if id == 'cow':
            animais['vaca'].append([*start, *end])
            cv2.rectangle(img, start, end, (255, 0, 0), 2)
        elif id == 'horse':
            lobos['min_x'].append(start[0])
            lobos['min_y'].append(start[1])
            lobos['max_x'].append(end[0])
            lobos['max_y'].append(end[1])
    

    start = (min(lobos['min_x']), min(lobos['min_y']))
    end = (max(lobos['max_x']), max(lobos['max_y']))
    animais['lobo'].append([*start, *end])
    cv2.rectangle(img, start, end, (0, 0, 255), 2)

    return img, animais


def calcula_iou(boxA, boxB):
    """Não mude ou renomeie esta função
        Calcula o valor do "Intersection over Union" para saber se as caixa se encontram
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def checar_perigo(image, animais):
    """Não mude ou renomeie esta função
        Recebe as coordenadas das caixas, se a caixa de uma vaca tem intersecção com as do lobo, ela esta em perigo.
        Se estiver em perigo, deve escrever na imagem com a cor vermlha, se não, escreva com a cor azul.
        *Importante*: nesta função, não faça cópia da imagem de entrada!!
        
        Repita para cada vaca na imagem.
    """
    for vaca in animais['vaca']:
        iou = calcula_iou(vaca, animais['lobo'][0])
        x, y, *_ = vaca
        if iou > 0.01:
            cv2.putText(image, f'PERIGO', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])
        else:
            cv2.putText(image, f'TRANQUILO', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 0, 0])
