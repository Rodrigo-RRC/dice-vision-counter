# importa o OpenCV. 
# É a biblioteca de visão computacional que vai fazer o trabalho 
# pesado: carregar imagens, converter cores, detectar círculos.
import cv2

# Toda imagem no OpenCV é na prática um ARRAY NumPy 
#  — uma grade de números. Vamos precisar dele (Numpy) 
# para manipular esses arrays.
import numpy as np


# importa o Optional do módulo de tipagem do Python. 
# Vou usar nos type hints quando uma função pode retornar 
# None (por exemplo, se a imagem não carregar).
from typing import Optional

# Limites do intervalo de matiz (Hue) para a cor roxa no espaço HSV.
# No OpenCV, o canal H vai de 0 a 180 (metade do círculo cromático de 360°).
# O roxo/violeta fica aproximadamente entre 125 e 155.
PURPLE_HUE_LOW  = np.array([130, 50, 50],  dtype=np.uint8)
PURPLE_HUE_HIGH = np.array([160, 255, 255], dtype=np.uint8)

def load_image(image_path:str)-> Optional[np.ndarray]:
    """
    Carrega uma imagem do disco e verifica se foi lida corretamente.

    Args:
      image_path: Caminho absoluto ou relativo da image_path

    Returns:
      A imagem como um arrqy numpy em formato BGR ou None se o 
      caminho não for encontrado
    """
    # o cv2.read() tenta abrir a imagem no caminho passado como
    # argumento. Caso o o caminho for inválido ou inexistente
    #  retorna None silenciosamente.

    image = cv2.imread(image_path)

    # verifico o None explicitamente para não deixar o erro se propagar 
    # pelo meu código
    
    if image is None:
        print(f"[ERRO] Não foi possível carregar a imagem: {image_path}")
        return None
    return image

    

def preprocess_image(image: np.ndarray)-> np.ndarray:
    """
    Converte a imagem do espaço de cor BGR para HSV.

    Trabalhamos em HSV porque ele separa a informação de matiz (qual cor)
    da informação de brilho (quão clara/escura), tornando a segmentação
    por cor muito mais robusta sob variações de iluminação.

    Args:
        image: Imagem original em formato BGR.

    Returns:
        A mesma imagem convertida para o espaço HSV.
    """
        # cv2.cvtColor converte entre espaços de cor.
        # O flag cv2.COLOR_BGR2HSV instrui o OpenCV a fazer a conversão BGR → HSV.

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    return hsv_image


def create_purple_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Cria uma máscara binária isolando os pixels dentro do intervalo de cor roxa.

    Uma máscara binária é uma imagem em preto e branco onde:
    - Branco (255) = pixel está dentro do intervalo de cor definido
    - Preto  (0)   = pixel está fora do intervalo

    Args:
        hsv_image: Imagem no espaço HSV.

    Returns:
        Máscara binária com os pixels roxos marcados em branco.
    """
    # cv2.inRange compara cada pixel da hsv_image contra os limites.
    # Pixels dentro do intervalo [PURPLE_HUE_LOW, PURPLE_HUE_HIGH] → 255 (branco)
    # Pixels fora do intervalo → 0 (preto)
    mask = cv2.inRange(hsv_image, PURPLE_HUE_LOW, PURPLE_HUE_HIGH)

    # Operação morfológica de fechamento (closing): dilatar seguido de erosão.
    # Isso preenche pequenos buracos dentro da região roxa causados
    # pela translucidez dos dados ou por reflexos de luz.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_dice_contours(mask: np.ndarray) -> list:
    """
    Encontra os contornos externos das regiões roxas na máscara,
    que correspondem aos dados presentes na imagem.

    Contornos pequenos (ruído) são filtrados por área mínima.

    Args:
        mask: Máscara binária com os pixels dos dados em branco.

    Returns:
        Lista de contornos válidos, cada um representando um dado.
    """
    # cv2.findContours detecta as bordas das regiões brancas na máscara.
    # RETR_EXTERNAL: recupera apenas os contornos mais externos (ignora hierarquia interna).
    # CHAIN_APPROX_SIMPLE: comprime segmentos de linha, guardando só os vértices.
    # contours = LISTA de contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtramos contornos por área mínima para eliminar ruído.
    # Pequenas manchas roxas que não sejam dados teriam área muito pequena.
    # O valor 500 pixels² é um limiar empírico para este conjunto de imagens.
    MIN_CONTOUR_AREA = 500
    valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    return valid_contours


def count_pips(image: np.ndarray, contour: np.ndarray) -> int:
    """
    Conta os pontos (pips) brancos na face visível de um dado.

    Para isso, recortamos a região do dado (ROI), convertemos para
    escala de cinza e aplicamos o detector de círculos de Hough.

    Args:
        image: Imagem original em BGR.
        contour: Contorno do dado dentro do qual procurar os pips.

    Returns:
        Número inteiro representando o valor da face do dado (1 a 6).
    """
    # cv2.boundingRect calcula o menor retângulo alinhado aos eixos
    # que envolve completamente o contorno.
    # Retorna: x e y do canto superior esquerdo, largura e altura.
    x, y, w, h = cv2.boundingRect(contour)

    # Recortamos apenas a região do dado da imagem original.
    # Isso é chamado de ROI (Region of Interest — Região de Interesse).
    # A notação NumPy image[y:y+h, x:x+w] significa:
    # "linhas de y até y+h, colunas de x até x+w"
    roi = image[y:y+h, x:x+w]

    # Convertemos o ROI para escala de cinza.
    # HoughCircles trabalha com imagem de canal único.
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Suavização Gaussiana para reduzir ruído antes da detecção de círculos.
    # O kernel (5,5) e sigma 2 são parâmetros de suavização.
    # Sem isso, HoughCircles tende a detectar círculos falsos em bordas ruidosas.
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 2)

    # cv2.HoughCircles detecta círculos usando a Transformada de Hough.
    # HOUGH_GRADIENT: método baseado em gradiente de intensidade.
    # dp=1.2: resolução do acumulador (1.0 = mesma resolução da imagem).
    # minDist: distância mínima entre centros de círculos detectados.
    #          Usamos h//4 para evitar detectar o mesmo pip duas vezes.
    # param1: limiar superior do detector de bordas Canny interno.
    # param2: limiar de votos no acumulador (menor = detecta mais, mas com mais falsos).
    # minRadius e maxRadius: tamanho esperado dos pips em pixels.
    circles = cv2.HoughCircles(
        roi_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h // 4,
        param1=50,
        param2=15,
        minRadius=5,
        maxRadius=30,
    )

    # HoughCircles retorna None se não encontrar nenhum círculo.
    if circles is None:
        return 0

    # circles tem shape (1, N, 3): N círculos, cada um com (x_centro, y_centro, raio).
    # np.round arredonda os valores e uint16 converte para inteiro sem sinal.
    circles = np.round(circles[0]).astype(np.uint16)

    return len(circles)


def detect_dice(image_path: str) -> int:
    """
    Pipeline completo de detecção de dados e soma das faces.

    Orquestra todas as etapas: carregamento, pré-processamento,
    segmentação por cor, detecção de contornos e contagem de pips.

    Args:
        image_path: Caminho para o arquivo de imagem a ser processado.

    Returns:
        Soma total dos valores de todos os dados encontrados na imagem.
        Retorna 0 se a imagem não puder ser carregada.
    """
    # --- Etapa 1: Carregar a imagem ---
    image = load_image(image_path)
    if image is None:
        return 0

    # --- Etapa 2: Converter para HSV ---
    hsv = preprocess_image(image)

    # --- Etapa 3: Criar máscara roxa ---
    mask = create_purple_mask(hsv)

    # --- Etapa 4: Encontrar contornos dos dados ---
    contours = find_dice_contours(mask)

    print(f"[INFO] {len(contours)} dado(s) encontrado(s) na imagem.")

    # --- Etapa 5: Contar pips em cada dado e acumular o total ---
    total = 0
    for i, contour in enumerate(contours):
        pip_count = count_pips(image, contour)
        print(f"  → Dado {i + 1}: {pip_count} ponto(s)")
        total += pip_count

    print(f"[RESULTADO] Soma total das faces: {total}")

    return total