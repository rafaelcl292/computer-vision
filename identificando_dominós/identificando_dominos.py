#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from ast import Break 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())


    
video = "dominoes.mp4"


if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            sys.exit(0)

        # Our operations on the frame come here

        # Criado mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 150, 255)

        kernel = np.ones([9, 9])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Recortando dominó
        y, x = np.nonzero(mask)
        domino_recortado = mask[np.min(y) : np.max(y), np.min(x) : np.max(x)]

        # Dividindo dominó em duas partes
        metade = domino_recortado.shape[0] // 2
        upper_domino = domino_recortado[:metade, :]
        lower_domino = domino_recortado[metade:, :]
        
        # Estimando tamanho dos círculos com base no tamanho do dominó
        largura_domino = domino_recortado.shape[1]
        min_radious = round(largura_domino / 9.6)
        max_radious = round(largura_domino / 8.3)

        # Detectando círculos em cada parte do dominó
        circles_in_upper_domino = cv2.HoughCircles(upper_domino, cv2.HOUGH_GRADIENT, dp=1, minDist=max_radious, param1=110, param2=9, minRadius=min_radious, maxRadius=max_radious)
        circles_in_lower_domino = cv2.HoughCircles(lower_domino, cv2.HOUGH_GRADIENT, dp=1, minDist=max_radious, param1=110, param2=9, minRadius=min_radious, maxRadius=max_radious)

        # Escrevendo resultado na imagem
        cv2.putText(frame, f'{circles_in_upper_domino.shape[1]} por {circles_in_lower_domino.shape[1]}', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, [0, 0, 255])

        # Desenhando círculos detectados no frame
        # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=max_radious, param1=110, param2=9, minRadius=min_radious, maxRadius=max_radious)
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     for x, y, r in circles:
        #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
