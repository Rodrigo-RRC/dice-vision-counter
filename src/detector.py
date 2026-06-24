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
    Conta os pontos (pips) brancos na face visivel de um dado.

    Em vez de HoughCircles (que, nestes dados translucidos, detectava
    reflexos, quinas arredondadas e pontos de outras faces vistos atraves
    do corpo), isolamos cada pip como uma "mancha" (blob) clara e usamos
    o cv2.SimpleBlobDetector. Ele filtra as deteccoes por FORMA, mantendo
    so manchas redondas, cheias e bem proporcionadas -- o que descarta
    naturalmente a borda brilhante do dado e os reflexos alongados.

    Args:
        image: Imagem original em BGR.
        contour: Contorno do dado dentro do qual procurar os pips.

    Returns:
        Numero de pips encontrados na face (tipicamente de 1 a 6).
    """
    # Recorta a regiao do dado (ROI) a partir do retangulo do contorno.
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]

    # O detector trabalha em tons de cinza. A suavizacao leve evita que
    # granulado da textura vire mancha falsa.
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 1)

    # A area do dado serve de referencia: assim os limites de tamanho dos
    # pips ficam proporcionais ao tamanho do dado na imagem.
    die_area = w * h

    params = cv2.SimpleBlobDetector_Params()

    # Cor: procuramos manchas CLARAS (pips brancos), nao escuras.
    params.filterByColor = True
    params.blobColor = 255

    # Area: descarta ruido minusculo e manchas grandes demais (reflexo amplo).
    params.filterByArea = True
    params.minArea = die_area * 0.004
    params.maxArea = die_area * 0.06

    # Circularidade: 1.0 = circulo perfeito. Descarta formas irregulares.
    params.filterByCircularity = True
    params.minCircularity = 0.6

    # Inercia: o quanto a mancha e redonda x alongada. Baixo = alongada
    # (tipico da borda/reflexo) -> descartada.
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Convexidade: descarta manchas com reentrancias (uniao de borda + pip).
    params.filterByConvexity = True
    params.minConvexity = 0.7

    detector = cv2.SimpleBlobDetector_create(params)

    # detect() devolve um keypoint por mancha aceita; o total de pips
    # e simplesmente a quantidade de keypoints.
    keypoints = detector.detect(roi_gray)

    return len(keypoints)


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