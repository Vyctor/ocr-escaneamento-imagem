from matplotlib import pyplot as plt
import imutils
import cv2
import numpy as np


def stack_images(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(
                    lables[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth*c+10, eachImgHeight *
                                                d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def find_rectangle_contours(contours):
    rectangle_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            perimeter = cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximate) == 4:
                rectangle_contours.append(contour)
    rectangle_contours = sorted(
        rectangle_contours, key=cv2.contourArea, reverse=True)

    return rectangle_contours


def get_corner_points(contour):
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return approximation


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def split_boxes(image):
    rows = np.vsplit(image, 5)
    boxes = []
    for row in rows:
        columns = np.hsplit(row, 5)
        for box in columns:
            boxes.append(box)
    return boxes


def show_answers(image, my_index, grading, answers, number_of_questions, number_of_choices):
    green = (0, 255, 0)
    red = (0, 0, 255)

    sec_w = int(image.shape[1] / number_of_questions)
    sec_h = int(image.shape[0] / number_of_choices)

    for x in range(0, number_of_questions):
        my_answer = my_index[x]
        c_x = (my_answer * sec_w) + sec_w // 2
        c_y = (x * sec_h) + sec_h // 2

        if grading[x] == 1:
            my_color = green
        else:
            my_color = red
            correct_answer = answers[x]
            cv2.circle(image, ((correct_answer*sec_w) + sec_w //
                       2, (x * sec_h) + sec_h // 2), 50, green, 10,  cv2.BORDER_WRAP)

        cv2.circle(image, (c_x, c_y), 50, my_color, 10,  cv2.BORDER_WRAP)

    return image


config_tesseract = "--tessdata-dir tessdata"


def mostrar(image):
    figura = plt.gcf()
    figura.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def encontrar_contornos(imagem):
    contornos = cv2.findContours(
        imagem, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = imutils.grab_contours(contornos)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:6]
    return contornos


def transformar_imagem(nome_imagem):
    imagem = cv2.imread(nome_imagem)
    imagem_original = imagem.copy()

    mostrar(imagem)

    (altura_imagem, largura_imagem) = imagem.shape[:2]

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 60, 160)
    mostrar(edged)

    contornos = encontrar_contornos(edged.copy())

    for c in contornos:
        peri = cv2.arcLength(c, True)
        aproximacao = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(aproximacao) == 4:
            maior = aproximacao
            break

    cv2.drawContours(imagem, maior, -1, (120, 255, 0), 28)
    cv2.drawContours(imagem, [maior], -1, (120, 255, 0), 2)
    mostrar(imagem)

    pontos_maior = ordenar_pontos(maior)
    pts1 = np.float32(pontos_maior)
    pts2 = np.float32([[0, 0], [largura_imagem, 0], [
                      largura_imagem, altura_imagem], [0, altura_imagem]])

    matriz = cv2.getPerspectiveTransform(pts1, pts2)
    transform = cv2.warpPerspective(
        imagem_original, matriz, (largura_imagem, altura_imagem))

    mostrar(transform)
    return transform


def processamento_imagem(imagem):
    img_process = cv2.cvtColor(cv2.resize(
        imagem, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    img_process = cv2.adaptiveThreshold(
        img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
    return img_process


def ordenar_pontos(pontos):
    pontos = pontos.reshape((4, 2))
    pontos_novos = np.zeros((4, 1, 2), dtype=np.int32)

    add = pontos.sum(1)
    pontos_novos[0] = pontos[np.argmin(add)]
    pontos_novos[2] = pontos[np.argmax(add)]

    diferenca = np.diff(pontos, axis=1)
    pontos_novos[1] = pontos[np.argmin(diferenca)]
    pontos_novos[3] = pontos[np.argmax(diferenca)]

    return pontos_novos


def transformacao_de_perspectiva(pontos_ordenados, largura_imagem, altura_imagem, imagem):
    pontos1 = np.float32(pontos_ordenados)
    pontos2 = np.float32([[0, 0], [largura_imagem, 0], [
                         largura_imagem, altura_imagem], [0, altura_imagem]])
    matriz = cv2.getPerspectiveTransform(pontos1, pontos2)
    transform = cv2.warpPerspective(
        imagem, matriz, (largura_imagem, altura_imagem))
    return transform
