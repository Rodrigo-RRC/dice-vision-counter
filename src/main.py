"""
Ponto de entrada do dice-vision-counter.

Este modulo orquestra a execucao do sistema de visao computacional:
varre o diretorio de imagens, executa o pipeline de deteccao
(definido em detector.py) para cada imagem encontrada e exibe,
ao final, um relatorio com a soma das faces de cada imagem.
"""

# pathlib oferece uma forma orientada a objetos de lidar com caminhos
# de arquivos, independente do sistema operacional (Windows, Linux, macOS).
from pathlib import Path

# Importamos a funcao de alto nivel do nosso modulo de deteccao.
# detect_dice() encapsula todo o pipeline e retorna a soma das faces
# dos dados encontrados em UMA imagem.
from detector import detect_dice


# Extensoes de arquivo de imagem que o programa deve considerar.
# Usamos um conjunto (set) porque a verificacao de pertencimento e
# muito rapida, e escrevemos em minusculas para comparar sem
# diferenciar maiusculas/minusculas (.JPG e .jpg sao tratadas igual).
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Caminho para a pasta com as imagens de entrada.
# Construimos o caminho a partir da localizacao deste proprio arquivo,
# para que o programa funcione de onde quer que seja executado:
#   Path(__file__)   -> caminho deste arquivo (src/main.py)
#   .resolve()       -> transforma em caminho absoluto
#   .parent          -> sobe de main.py para a pasta src/
#   .parent          -> sobe de src/ para a raiz do projeto
#   / "data" / "raw" -> entra em data/raw a partir da raiz
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def find_image_paths(directory: Path) -> list[Path]:
    """
    Varre um diretorio e retorna os caminhos das imagens validas.

    Args:
        directory: Pasta a ser varrida em busca de imagens.

    Returns:
        Lista ordenada de caminhos (Path) cujas extensoes constam
        em VALID_EXTENSIONS. Lista vazia se a pasta nao existir.
    """
    # Se a pasta nao existe, avisamos e retornamos lista vazia em vez
    # de deixar um erro estourar mais adiante no programa.
    if not directory.is_dir():
        print(f"[ERRO] Diretorio nao encontrado: {directory}")
        return []

    # directory.iterdir() percorre todos os itens dentro da pasta.
    # Mantemos apenas os itens que:
    #   - sao arquivos (is_file), descartando eventuais subpastas;
    #   - tem extensao (em minusculas) presente em VALID_EXTENSIONS.
    # sorted() garante ordem alfabetica deterministica (img1, img2, ...).
    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    return image_paths


def main() -> None:
    """
    Funcao principal: encontra as imagens, executa a deteccao em cada
    uma e imprime um relatorio consolidado com os resultados.
    """
    # Etapa 1: descobrir quais imagens processar.
    image_paths = find_image_paths(DATA_DIR)

    # Se nenhuma imagem foi encontrada, nao ha o que processar.
    if not image_paths:
        print("[AVISO] Nenhuma imagem encontrada para processar.")
        return

    print(f"[INFO] {len(image_paths)} imagem(ns) encontrada(s) em {DATA_DIR}\n")

    # Dicionario para guardar o resultado de cada imagem.
    # Chave = nome do arquivo; valor = soma das faces retornada por detect_dice.
    results: dict[str, int] = {}

    # Etapa 2: processar cada imagem individualmente.
    for image_path in image_paths:
        print(f"=== Processando: {image_path.name} ===")

        # detect_dice espera o caminho como string, entao convertemos o Path.
        total = detect_dice(str(image_path))
        results[image_path.name] = total
        print()  # linha em branco para separar visualmente as imagens

    # Etapa 3: relatorio final consolidado.
    print("=" * 40)
    print("RELATORIO FINAL")
    print("=" * 40)
    for filename, total in results.items():
        print(f"  {filename}: soma das faces = {total}")


# Este bloco garante que main() so seja executado quando o arquivo for
# rodado diretamente (python main.py), e nao quando for importado por
# outro modulo. E uma convencao padrao em Python.
if __name__ == "__main__":
    main()