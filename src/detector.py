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
    Conta os pontos (pips) brancos na face de cima de um dado.

    O DESAFIO: os dados são de plástico TRANSLÚCIDO. Olhando de cima,
    enxergamos não só os pontos da face visível, mas também (mais fracos e
    "borrados") os pontos das outras faces vistos através do corpo do dado
    -- os "fantasmas". Contar apenas regiões claras em tons de cinza faz o
    programa somar esses fantasmas por engano.

    A ESTRATÉGIA usa dois fundamentos que valem em escalas/iluminações
    diferentes, no lugar de um limiar de brilho fixo:

      1) White top-hat morfológico: realça só manchas CLARAS e PEQUENAS
         (do tamanho de um pip), apagando o corpo do dado e reflexos
         grandes. O elemento estruturante é proporcional ao tamanho do
         dado, então a técnica se adapta a dados maiores/menores.

      2) Porta de "brancura" pela saturação: o ponto da face de cima é
         branco (baixa saturação no HSV); o fantasma chega tingido de roxo
         (alta saturação). Filtrar por saturação baixa separa, pela física
         da cor, o pip real do fantasma.

    O limiar final é via Otsu (corte escolhido automaticamente a partir da
    própria imagem), evitando número mágico de brilho.

    Args:
        image: Imagem original em BGR.
        contour: Contorno do dado dentro do qual procurar os pips.

    Returns:
        Numero de pips encontrados na face (tipicamente de 1 a 6).
    """
    # Recorta a regiao do dado (ROI) a partir do retangulo do contorno.
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]

    # Trabalhamos no espaco HSV: o canal V (brilho) acha os pontos claros;
    # o canal S (saturacao) distingue o branco real do fantasma roxo.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]        # V: brilho
    saturation = hsv[:, :, 1]   # S: saturacao

    # --- Fundamento 1: White top-hat ---
    # Kernel com ~1/3 do menor lado do dado: grande o bastante para o
    # top-hat "apagar" o corpo do dado e sobrar so os pips. O "| 1" forca
    # tamanho IMPAR (exigencia do elemento estruturante).
    pip_kernel_size = max(3, int(min(w, h) * 0.33)) | 1
    pip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (pip_kernel_size, pip_kernel_size))
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, pip_kernel)

    # Limiar de Otsu: separa automaticamente os pips realcados do fundo,
    # sem depender de um valor fixo de brilho.
    _, bright_spots = cv2.threshold(
        tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Fundamento 2: Porta de brancura ---
    # Mantemos so pixels de baixa saturacao (brancos). Os fantasmas roxos,
    # por terem saturacao alta, sao descartados aqui.
    white_gate = cv2.inRange(saturation, 0, 110)
    pip_mask = cv2.bitwise_and(bright_spots, white_gate)

    # Abertura morfologica (open) para remover respingos isolados de ruido.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, small_kernel)

    # A area do dado serve de referencia: assim os limites de tamanho dos
    # pips ficam proporcionais ao tamanho do dado na imagem.
    die_area = w * h

    params = cv2.SimpleBlobDetector_Params()

    # Cor: procuramos manchas CLARAS (ja sao brancas na mascara binaria).
    params.filterByColor = True
    params.blobColor = 255

    # Area: descarta ruido minusculo e manchas grandes demais (reflexo amplo).
    # Limites um pouco mais folgados porque a mascara ja chega limpa (so branco).
    params.filterByArea = True
    params.minArea = die_area * 0.003
    params.maxArea = die_area * 0.12

    # Circularidade: 1.0 = circulo perfeito. Descarta formas irregulares.
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Inercia: o quanto a mancha e redonda x alongada. Baixo = alongada
    # (tipico da borda/reflexo) -> descartada.
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Convexidade: descarta manchas com reentrancias (uniao de borda + pip).
    params.filterByConvexity = True
    params.minConvexity = 0.6

    detector = cv2.SimpleBlobDetector_create(params)

    # detect() devolve um keypoint por mancha aceita; o total de pips
    # e simplesmente a quantidade de keypoints. Agora rodamos sobre a
    # mascara binaria limpa (pip_mask), nao sobre o cinza bruto.
    keypoints = detector.detect(pip_mask)

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