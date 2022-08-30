#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import numpy as np
import math

def encontrar_foco(D, H, h):
    """Não mude ou renomeie esta função
    Entradas:
       D - distancia real da câmera até o objeto (papel)
       H - a distancia real entre os circulos (no papel)
       h - a distancia na imagem entre os circulos
    Saída:
       f - a distância focal da câmera
    """
    f = D * h / H

    return f


def segmenta_circulo_ciano(hsv): 
    """Não mude ou renomeie esta função
    Entrada:
        hsv - imagem em hsv
    Saída:
        mask - imagem em grayscale com tudo em preto e os pixels do circulos ciano em branco
    """

    menor = (int(170/2), 120, 70)
    maior = (int(220/2), 255, 255)
    mask = cv2.inRange(hsv, menor, maior)
    
    return mask


def segmenta_circulo_magenta(hsv):
    """Não mude ou renomeie esta função
    Entrada:
        hsv - imagem em hsv
    Saída:
        mask - imagem em grayscale com tudo em preto e os pixels do circulos magenta em branco
    """
    menor = (int(280/2), 120, 100)
    maior = (int(360/2), 255, 255)
    mask = cv2.inRange(hsv, menor, maior)
    
    return mask


def encontrar_maior_contorno(segmentado):
    """Não mude ou renomeie esta função
    Entrada:
        segmentado - imagem em preto e branco
    Saída:
        contorno - maior contorno obtido (APENAS este contorno)
    """

    contornos, _ = cv2.findContours(segmentado, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contorno = max(contornos, key=len)
    return contorno


def encontrar_centro_contorno(contorno):
    """Não mude ou renomeie esta função
    Entrada:
        contorno: um contorno (não o array deles)
    Saída:
        (Xcentro, Ycentro) - uma tuple com o centro do contorno (no formato 'int')!!! 
    """

    Xcentro, Ycentro = contorno.mean(axis=0)[0].astype(int)
    return (Xcentro, Ycentro)


def calcular_h(centro_ciano, centro_magenta):
    """Não mude ou renomeie esta função
    Entradas:
        centro_ciano - ponto no formato (X,Y)
        centro_magenta - ponto no formato (X,Y)
    Saída:
        distancia - a distancia Euclidiana entre os pontos de entrada 
    """
    
    distancia = (
        (centro_ciano[0] - centro_magenta[0])**2 + (centro_ciano[1] - centro_magenta[1])**2
    )**0.5
    
    return distancia


def encontrar_distancia(f, H, h):
    """Não mude ou renomeie esta função
    Entrada:
        f - a distância focal da câmera
        H - A distância real entre os pontos no papel
        h - a distância entre os pontos na imagem
    Saída:
        D - a distância do papel até câmera
    """

    D = f * H / h
    return D


def calcular_distancia_entre_circulos(img):
    """Não mude ou renomeie esta função
    Deve utilizar as funções acima para calcular a distancia entre os circulos a partir da imagem BGR
    Entradas:
        img - uma imagem no formato BGR
    Saídas:
        h - a distância entre os os circulos na imagem
        centro ciano - o centro do círculo ciano no formato (X,Y)
        centro_magenta - o centro do círculo magenta no formato (X,Y)
        img_contornos - a imagem com os contornos desenhados
    """
    img_contornos = img.copy()

    hsv = cv2.cvtColor(img_contornos, cv2.COLOR_BGR2HSV)
    
    ciano_segmentado = segmenta_circulo_ciano(hsv)
    magenta_segmentado = segmenta_circulo_magenta(hsv)

    contorno_ciano = encontrar_maior_contorno(ciano_segmentado)
    centro_ciano = encontrar_centro_contorno(contorno_ciano)

    contorno_magenta = encontrar_maior_contorno(magenta_segmentado)
    centro_magenta = encontrar_centro_contorno(contorno_magenta)

    h = calcular_h(centro_ciano, centro_magenta)

    cv2.drawContours(img_contornos, contorno_magenta, -1, [255, 0, 255], 5)
    cv2.drawContours(img_contornos, contorno_ciano, -1, [255, 255, 0], 5)
    
    return h, centro_ciano, centro_magenta, img_contornos


def calcular_angulo_com_horizontal_da_imagem(centro_ciano, centro_magenta):
    """Não mude ou renomeie esta função
        Deve calcular o angulo, em graus, entre o vetor formato com os centros do circulos e a horizontal.
    Entradas:
        centro_ciano - centro do círculo ciano no formato (X,Y)
        centro_magenta - centro do círculo magenta no formato (X,Y)
    Saídas:
        angulo - o ângulo entre os pontos em graus
    """

    xc, yc = (centro_ciano[0] - centro_magenta[0], centro_ciano[1] - centro_magenta[1])

    xh, yh = (1, 0)
    
    angulo = math.degrees(
        math.acos(
            (xc*xh + yc*yh) / ((xc**2 + yc**2)**0.5 * (xh**2 + yh**2)**0.5)
        )
    )

    return angulo
