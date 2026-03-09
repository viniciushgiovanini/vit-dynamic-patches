"""
Script para gerar máscaras de imagens em GPU para processamento mais rápido.
Baseado em: notebook/Mask/mask_genertion.ipynb

Estrutura:
- Lê imagens de: data/base_recortada/{classe}/
- Salva máscaras em: data/base_mascara/{classe}/
"""

import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Verificar disponibilidade de GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando device: {DEVICE}")


def listar_imagens(diretorio_raiz, extensoes_imagens=None):
    """
    Lista todas as imagens em um diretório recursivamente.

    Args:
        diretorio_raiz: Caminho raiz do diretório
        extensoes_imagens: Lista de extensões para filtrar

    Returns:
        Lista de caminhos completos das imagens
    """
    if extensoes_imagens is None:
        extensoes_imagens = [".jpg", ".jpeg", ".png"]

    arquivos_imagens = []
    for dirpath, _, filenames in os.walk(diretorio_raiz):
        for filename in filenames:
            if any(
                filename.lower().endswith(extensao)
                for extensao in extensoes_imagens
            ):
                caminho_completo = os.path.join(dirpath, filename)
                arquivos_imagens.append(caminho_completo)
    return arquivos_imagens


def remover_fundo_com_grabcut_recortado(imagem):
    """
    Remove fundo da imagem usando GrabCut.

    Args:
        imagem: Imagem BGR

    Returns:
        Tupla (imagem_sem_fundo, mascara, imagem_original_recortada)
    """
    mascara = np.zeros(imagem.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    altura, largura = imagem.shape[:2]

    # Calcular os limites do retângulo
    x1 = 0
    y1 = 0
    x2 = largura - 1
    y2 = altura - 1

    rectangle = (x1, y1, x2 - x1, y2 - y1)

    cv2.grabCut(
        imagem,
        mascara,
        rectangle,
        backgroundModel,
        foregroundModel,
        3,
        cv2.GC_INIT_WITH_RECT,
    )

    mascara_objeto = np.where((mascara == 2) | (mascara == 0), 0, 1).astype(
        "uint8"
    )

    imagem_sem_fundo = imagem * mascara_objeto[:, :, np.newaxis]

    img_recortada = imagem_sem_fundo[y1:y2, x1:x2]

    imagem_gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

    _, mascara = cv2.threshold(imagem_gray, 10, 255, cv2.THRESH_BINARY)

    img_original_recortada = imagem[y1:y2, x1:x2]

    return img_recortada, mascara, img_original_recortada


def processar_imagem(caminho_imagem, diretorio_saida_mascara):
    """
    Processa uma única imagem gerando sua máscara.

    Args:
        caminho_imagem: Caminho completo da imagem
        diretorio_saida_mascara: Diretório onde salvar a máscara

    Returns:
        Tupla (sucesso, mensagem)
    """
    try:
        # Ler imagem
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            return False, f"Erro ao ler: {caminho_imagem}"

        # Obter informações do caminho
        partes_caminho = caminho_imagem.split(os.sep)
        classe = partes_caminho[-2]  # Nome da classe (pasta pai)
        nome_arquivo = os.path.basename(caminho_imagem)

        # Gerar máscara
        imagem_sem_fundo, mascara, img_original = (
            remover_fundo_com_grabcut_recortado(imagem)
        )

        # Criar diretório da classe se não existir
        diretorio_classe = os.path.join(diretorio_saida_mascara, classe)
        if not os.path.exists(diretorio_classe):
            os.makedirs(diretorio_classe, exist_ok=True)

        # Salvar máscara
        if len(mascara) != 0:
            caminho_saida = os.path.join(diretorio_classe, nome_arquivo)
            cv2.imwrite(caminho_saida, mascara)
            return True, f"Processada: {classe}/{nome_arquivo}"
        else:
            return False, f"Máscara vazia: {classe}/{nome_arquivo}"

    except Exception as e:
        return False, f"Erro ao processar {caminho_imagem}: {str(e)}"


def processar_em_paralelo(
    imagens_caminhos, diretorio_saida_mascara, num_workers=None
):
    """
    Processa imagens em paralelo usando múltiplas threads.

    Args:
        imagens_caminhos: Lista de caminhos de imagens
        diretorio_saida_mascara: Diretório de saída
        num_workers: Número de workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(
        f"Processando {len(imagens_caminhos)} imagens com {num_workers} workers"
    )

    sucessos = 0
    falhas = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(processar_imagem, img_path, diretorio_saida_mascara)
            for img_path in imagens_caminhos
        ]

        for future in tqdm(futures, total=len(futures), desc="Processando"):
            try:
                sucesso, mensagem = future.result()
                if sucesso:
                    sucessos += 1
                else:
                    falhas += 1
                    logger.warning(mensagem)
            except Exception as e:
                falhas += 1
                logger.error(f"Erro na thread: {str(e)}")

    logger.info(
        f"Processamento concluído: {sucessos} sucessos, {falhas} falhas"
    )
    return sucessos, falhas


def gerar_mascaras_gpu(
    diretorio_entrada="data/base_recortada/",
    diretorio_saida="data/base_mascara/",
    num_workers=None,
):
    """
    Função principal para gerar máscaras em GPU/paralelo.

    Args:
        diretorio_entrada: Diretório com imagens organizadas por classe
        diretorio_saida: Diretório de saída para máscaras
        num_workers: Número de workers para processamento paralelo
    """
    # Validar diretórios de entrada
    if not os.path.exists(diretorio_entrada):
        logger.error(f"Diretório de entrada não existe: {diretorio_entrada}")
        return False

    # Criar diretório de saída
    os.makedirs(diretorio_saida, exist_ok=True)

    # Listar todas as imagens
    logger.info(f"Listando imagens em: {diretorio_entrada}")
    imagens = listar_imagens(diretorio_entrada)
    logger.info(f"Total de imagens encontradas: {len(imagens)}")

    if len(imagens) == 0:
        logger.warning("Nenhuma imagem encontrada!")
        return False

    # Processar em paralelo
    sucessos, falhas = processar_em_paralelo(
        imagens, diretorio_saida, num_workers
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Geração de máscaras concluída com sucesso!")
    logger.info(f"Total processado: {sucessos + falhas}")
    logger.info(f"Sucessos: {sucessos}")
    logger.info(f"Falhas: {falhas}")
    logger.info(f"{'='*50}")

    return True


def verificar_mascaras_geradas(
    diretorio_entrada="data/base_recortada/",
    diretorio_saida="data/base_mascara/",
):
    """
    Verifica quantas máscaras foram geradas por classe.

    Args:
        diretorio_entrada: Diretório de imagens de entrada
        diretorio_saida: Diretório de máscaras geradas
    """
    logger.info("\nVerificação de máscaras geradas:")
    logger.info(f"{'Classe':<40} {'Entrada':<10} {'Saída':<10} {'Status':<10}")
    logger.info("-" * 70)

    # Listar classes (subdiretórios)
    classes = [
        d
        for d in os.listdir(diretorio_entrada)
        if os.path.isdir(os.path.join(diretorio_entrada, d))
    ]

    total_entrada = 0
    total_saida = 0

    for classe in sorted(classes):
        dir_classe_entrada = os.path.join(diretorio_entrada, classe)
        dir_classe_saida = os.path.join(diretorio_saida, classe)

        # Contar imagens de entrada
        imagens_entrada = [
            f
            for f in os.listdir(dir_classe_entrada)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        num_entrada = len(imagens_entrada)

        # Contar máscaras de saída
        if os.path.exists(dir_classe_saida):
            mascaras_saida = [
                f
                for f in os.listdir(dir_classe_saida)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            num_saida = len(mascaras_saida)
        else:
            num_saida = 0

        total_entrada += num_entrada
        total_saida += num_saida

        status = "✓ Completo" if num_entrada == num_saida else "✗ Incompleto"
        logger.info(
            f"{classe:<40} {num_entrada:<10} {num_saida:<10} {status:<10}"
        )

    logger.info("-" * 70)
    logger.info(f"{'TOTAL':<40} {total_entrada:<10} {total_saida:<10}")
    logger.info(
        f"Percentual completo: {(total_saida/total_entrada*100 if total_entrada > 0 else 0):.1f}%\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gera máscaras de imagens em GPU/paralelo"
    )
    parser.add_argument(
        "--entrada",
        default="../../data/base_recortada",
        help="Diretório de entrada com imagens",
    )
    parser.add_argument(
        "--saida",
        default="../../data/base_mascara/",
        help="Diretório de saída para máscaras",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--verificar",
        action="store_true",
        help="Apenas verificar máscaras geradas",
    )

    args = parser.parse_args()

    if args.verificar:
        verificar_mascaras_geradas(args.entrada, args.saida)
    else:
        gerar_mascaras_gpu(args.entrada, args.saida, args.workers)
        verificar_mascaras_geradas(args.entrada, args.saida)
        verificar_mascaras_geradas(args.entrada, args.saida)
