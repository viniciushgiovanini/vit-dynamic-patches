import os
import pandas as pd
import cv2
import shutil
import random
import albumentations as A
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm
import pickle


class CustomImageFolder(ImageFolder):
    """
        Sobrescreve o método __getitem__ da classe base ImageFolder.

        Este método recupera a imagem, o rótulo (target) e o nome do arquivo de imagem
        correspondente.

        Args:
            index (int): Índice do item a ser recuperado.

        Returns:
            tuple: Uma tupla contendo:
                - img (PIL.Image.Image): A imagem carregada.
                - target (int): O rótulo associado à imagem.
                - image_name (str): O nome do arquivo da imagem.
        """

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        path = self.imgs[index][0]

        image_name = os.path.basename(path)

        return img, target, image_name


class Preprocessamento():
    def __init__(self, root_path):
        self.root_path = root_path
        self.csv_path = self.root_path + "classifications.csv"

    def create_folders(self):
        """
        Cria as pastas necessários para o armazenamento de dados, caso ainda não existam.

        Returns:
            None
        """
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

        folders_paths = [
            f"{self.root_path}base/",
            f"{self.root_path}base_temporaria",
            f"{self.root_path}base_recortada",
            f"{self.root_path}base_mascara",
            f"{self.root_path}base_treinamento",
            f"{self.root_path}centros_pre_salvos",
            f"modelos/",
            f"figs/",
            f"graph/",
        ]

        for each in folders_paths:
            if not os.path.exists(each):
                os.mkdir(each)
                print(f'Dirétorio criado: ({each})')

    def clone_dataset(self):
        """
        Clona um conjunto de dados do diretório de origem para um diretório de destino.

        Returns:
            None
        """

        raiz = f"{self.root_path}/base"
        destino = f"{self.root_path}/base_temporaria"

        if ".png" in raiz:
            source_file = raiz

            raiz_split = raiz.split("/")

            destination_file = os.path.join(
                destino, raiz_split[(len(raiz_split)-1)])
            shutil.copy2(raiz, destination_file)
        else:
            for i, each_file in enumerate(os.listdir(raiz)):

                source_file = os.path.join(raiz, each_file)

                destination_file = os.path.join(destino, each_file)

                if os.path.isfile(source_file):
                    shutil.copy2(source_file, destination_file)

    def listar_imagens(self, diretorio_raiz, extensoes_imagens=None):
        """
        Lista todos os arquivos de imagem em um diretório e seus subdiretórios.

        Args:
            diretorio_raiz (str): O caminho do diretório raiz onde a busca pelas imagens sera realizada.
            extensoes_imagens (list, opcional): Uma lista de extensões de arquivos de imagem a serem incluídas na busca.

        Returns:
            list: Uma lista de strings com os caminhos completos dos arquivos de imagem encontrados.
        """

        if extensoes_imagens is None:
            extensoes_imagens = ['.jpg', '.jpeg', '.png']

        arquivos_imagens = []
        for dirpath, _, filenames in os.walk(diretorio_raiz):
            for filename in filenames:
                if any(filename.lower().endswith(extensao) for extensao in extensoes_imagens):
                    caminho_completo = os.path.join(dirpath, filename)
                    arquivos_imagens.append(caminho_completo)
        return arquivos_imagens

    def crop_dataset(self, value_expand=90):
        """
        Recorta as imagens do dataset com base nas coordenadas fornecidas do arquivo CSV.

        Args:
            value_expand (int, opcional): O valor de expansão ao redor da posição da célula para recorte da imagem.
                                          O padrão é 90 pixels (45 pixels para cada lado).

        Returns:
            None
        """

        diretorio_dataset = f"{self.root_path}base_temporaria/"
        diretorio_dataset_recortado = f"{self.root_path}base_recortada/"
        df = pd.read_csv(f"{self.root_path}classifications.csv")

        img_base_temporaria = self.listar_imagens(
            f"{self.root_path}/base_temporaria/")

        img_base_temporaria_img_name = []

        for each in img_base_temporaria:
            img_base_temporaria_img_name.append(
                each.replace(f"{self.root_path}/base_temporaria/", ""))

        df = df[df["image_filename"].isin(img_base_temporaria_img_name)]
        # df = df[df["bethesda_system"] == "ASC-H"]

        value_expand2 = int(value_expand / 2)

        print(value_expand2)

        for each in df.iterrows():

            nome_img = each[1]['image_filename']
            nome_da_doenca = each[1]['bethesda_system']
            posi_x = each[1]['nucleus_x']
            posi_y = each[1]['nucleus_y']
            id_celular = each[1]['cell_id']

            path_imagem_dataset_original = f'{diretorio_dataset}{nome_img}'
            print(path_imagem_dataset_original)

            # Onde ele vai ler cada imagem;
            img = cv2.imread(path_imagem_dataset_original)

            x1 = max(0, posi_x - value_expand2)
            y1 = max(0, posi_y - value_expand2)
            x2 = min(img.shape[1], posi_x + value_expand2)
            y2 = min(img.shape[0], posi_y + value_expand2)

            # Recortando a imagem;
            img_recortada = img[y1:y2, x1:x2]

            # Verifica se existe um folder no destino com o nome da doenca;
            if not os.path.exists(os.path.join(diretorio_dataset_recortado, nome_da_doenca)):
                os.mkdir(os.path.join(
                    diretorio_dataset_recortado, nome_da_doenca))

            # Salva a imagem recortada no novo destino

            # print(len(img_recortada))

            if (len(img_recortada) != 0):
                try:
                    cv2.imwrite(
                        f'{diretorio_dataset_recortado}{nome_da_doenca}/' + f'{id_celular}.png', img_recortada)
                except:
                    print(id_celular)

    def obter_nomes_pastas(self, diretorio):
        """
        Obtém os nomes das pastas presentes em um diretorio.

        Args:
            diretorio (str): O caminho do diretorio no qual as pastas serao listadas.

        Returns:
            list: Lista com os nomes das pastas encontradas no diretorio.
        """

        nomes_pastas = []
        for nome in os.listdir(diretorio):
            if os.path.isdir(os.path.join(diretorio, nome)):
                nomes_pastas.append(nome)
        return nomes_pastas

    def mover_imagens(self, origem, destino, quantidade):
        """
        Move imagens de um diretorio de origem para um diretorio de destino.

        Args:
            origem (str): O diretorio de origem contendo as imagens a serem movidas.
            destino (str): O diretorio de destino onde as imagens serao movidas.
            quantidade (int): O numero de imagens a serem movidas. Se houver menos imagens do que o
                              especificado, todas as imagens serao movidas.

        Returns:
            None
        """

        if not os.path.exists(destino):
            os.makedirs(destino)

        arquivos = self.listar_imagens(origem)

        if arquivos:
            arquivos_para_mover = random.sample(
                arquivos, min(quantidade, len(arquivos)))

            for arquivo in arquivos_para_mover:

                tmp = arquivo.split("/")

                img_name = tmp[-1]

                caminho_arquivo_origem = os.path.join(origem,  img_name)
                caminho_arquivo_destino = os.path.join(destino,  img_name)

                shutil.move(caminho_arquivo_origem, caminho_arquivo_destino)
        else:
            print(f'Nenhuma imagem encontrada na pasta: {origem}')

    def divisao_dataset(self, percent_train_dataset=0.8):
        """
        Divide o dataset em dois conjuntos: treinamento e validacao, com base no percentual fornecido.

        Args:
            percent_train_dataset (float): Percentual de imagens a serem usadas para treinamento (default é 0.8, ou 80%).

        Returns:
            None
        """

        cwd = os.getcwd()
        diretorio = self.root_path + "base_recortada"

        pastas = self.obter_nomes_pastas(diretorio)

        diretorio_origem = diretorio
        diretorio_destino = self.root_path + "base_treinamento"

        percent = percent_train_dataset

        for each in pastas:

            diretorio_imagens = diretorio_origem + f'/{each}/'

            diretorio_treinamento = diretorio_destino + f'/train/{each}/'

            diretorio_teste = diretorio_destino + f'/validation/{each}/'

            imagens = os.listdir(diretorio_imagens)

            random.shuffle(imagens)

            indice_divisao = int(percent * len(imagens))

            imagens_treinamento = imagens[:indice_divisao]
            imagens_teste = imagens[indice_divisao:]

            if not (os.path.exists(diretorio_teste)):
                os.makedirs(diretorio_teste)

            if not (os.path.exists(diretorio_treinamento)):
                os.makedirs(diretorio_treinamento)

            for imagem in imagens_treinamento:
                shutil.copy(os.path.join(diretorio_imagens, imagem),
                            diretorio_treinamento)

            for imagem in imagens_teste:
                shutil.copy(os.path.join(
                    diretorio_imagens, imagem), diretorio_teste)

        origem = f'{self.root_path}base_treinamento/train/'
        destino = f'{self.root_path}base_treinamento/test/'

        pastas = self.obter_nomes_pastas(diretorio)

        for each in pastas:
            path_final = origem + each + "/"

            destino_final = destino + each + "/"

            all_image = self.listar_imagens(path_final)

            quantidade_resultado = int((20 / 100) * len(all_image))

            print(
                f"O total de imagens de treino no path {path_final} são {len(path_final)}, e 20% destas são {quantidade_resultado}")

            self.mover_imagens(path_final, destino_final,
                               quantidade_resultado)

    def listaImg(self, path_list):
        """
        Lista todas as imagens em um conjunto de diretorios e retorna um dicionario com os caminhos completos das imagens.

        Args:
            path_list (list of str): Lista de diretorios contendo as imagens para listar.

        Returns:
            dict: Dicionario onde as chaves sao os nomes das pastas e os valores sao listas com os caminhos completos das imagens.
        """

        lista = {}

        for each in path_list:
            name = each.split("/")
            name = name[len(name)-1]
            all_image = [f for f in os.listdir(
                each) if os.path.isfile(os.path.join(each, f))]

            all_images_path_complete = []
            for each_img in all_image:
                all_images_path_complete.append(each + "/" + each_img)
            lista[name] = all_images_path_complete

            print(
                f"Quantidade de imagens no path: {name} é ----> {len(lista[name])}")
        return lista

    def myaugment(self, lista_img, qtd):
        """
        Aplica aumento de dados em imagens para gerar mais exemplos no dataset.

        Args:
            lista_img (list of str): Lista com os caminhos completos das imagens a serem aumentadas.
            qtd (int): Número total de imagens desejadas após o aumento. A função gera imagens até que o total de imagens atinja esse valor.

        Returns:
            None
        """

        qtd_img = qtd - len(lista_img)

        if qtd_img <= 0:
            print("Essa quantidade de dados já está presente")
            return

        for i in range(qtd_img):

            total_img = len(lista_img)

            img_aleatoria = random.randint(0, total_img-1)

            path = lista_img[img_aleatoria]

            id_da_imagem = path.split(
                "/")[-1].replace(".png", "").replace(".jpg", "").replace(".jpeg", "")

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            altura, largura = image.shape[:2]

            tamanho_desejado = (90, 90)

            if (largura, altura) != tamanho_desejado:
                image = cv2.resize(image, tamanho_desejado,
                                   interpolation=cv2.INTER_AREA)

            # plt.imshow(img)

            new_path = path.split("/")
            new_path.pop(len(new_path)-1)
            new_path = "/".join(new_path) + "/"

            # graus_random = random.randint(5,330)

            transform = A.Compose([
                A.RandomCrop(width=90, height=90),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
            ])
            # transform = A.Compose([
            #   A.RandomCrop(width=90, height=90),
            #   A.HorizontalFlip(p=0.5),
            #   A.ShiftScaleRotate(p=0.5),
            #   A.Perspective(p=0.5),
            #   A.Affine(p=0.7, rotate=graus_random, mode=cv2.BORDER_REPLICATE)
            # ])

            trf_image = transform(image=image)['image']

            trf_image = cv2.cvtColor(trf_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(
                new_path + f'augmentation_id_{id_da_imagem}_{i}.png', trf_image)
            i = i + 1

        print(
            f"Gerou: {i} imagens, com as {len(lista_img)} deu no total {len(lista_img) + i}")

    def aumento_de_dados(self, valor_aumentar=1000):
        """
        Realiza o aumento de dados em varias classes de imagens no conjunto de treinamento, gerando um numero especifico de novas imagens.

        Args:
            valor_aumentar (int): Quantidade total de imagens que devem existir na pasta para cada classe (default é 1000).

        Returns:
            None
        """

        train_or_validation = "train"

        path_list = [
            f"{self.root_path}base_treinamento/{train_or_validation}/ASC-H",
            f"{self.root_path}base_treinamento/{train_or_validation}/ASC-US",
            f"{self.root_path}base_treinamento/{train_or_validation}/LSIL",
            f"{self.root_path}base_treinamento/{train_or_validation}/HSIL",
            f"{self.root_path}base_treinamento/{train_or_validation}/Negative for intraepithelial lesion",
            f"{self.root_path}base_treinamento/{train_or_validation}/SCC",

        ]

        lista_img = self.listaImg(path_list)

        for each_class in path_list:
            class_name = each_class.split("/")[-1]
            print(class_name)
            self.myaugment(
                lista_img=lista_img[f"{class_name}"], qtd=valor_aumentar)

    def reducao_de_dados(self, valor_deletar=1000):
        valor_deletar = 1000
        train_or_validation = "train"

        path_list = [
            f"{self.root_path}base_treinamento/{train_or_validation}/ASC-H",
            f"{self.root_path}base_treinamento/{train_or_validation}/ASC-US",
            f"{self.root_path}base_treinamento/{train_or_validation}/LSIL",
            f"{self.root_path}base_treinamento/{train_or_validation}/HSIL",
            f"{self.root_path}base_treinamento/{train_or_validation}/Negative for intraepithelial lesion",
            f"{self.root_path}base_treinamento/{train_or_validation}/SCC",

        ]

        lista = {}

        for each in path_list:
            name = each.split("/")
            name = name[len(name)-1]
            all_image = [f for f in os.listdir(
                each) if os.path.isfile(os.path.join(each, f))]

            all_images_path_complete = []
            for each_img in all_image:
                all_images_path_complete.append(each + "/" + each_img)
            lista[name] = all_images_path_complete

            # print(f"Quantidade de imagens no path: {name} é ----> {len(lista[name])}")

        for k, v in lista.items():
            print(k)
            print(v)

            diff = len(v) - valor_deletar

            print(
                f"O Arquivo tem {len(v)} imagens e menos o {valor_deletar} = {diff}")

            if diff > 0:

                path_excluido = lista[k]
                random.shuffle(path_excluido)

                print(f"Fazendo a exclusão de {diff} imagens no {k} path")

                for i, each in enumerate(path_excluido):
                    if i == diff:
                        break
                    # print(each)
                    os.remove(each)


##############################################################################################
#  CLASSE PARA GERAÇÃO DA LISTA DE CENTROS PARA CADA ABORDAGEM DE EXTRAÇÃO DINAMICA DE PATCHES
##############################################################################################


class GeracaoListaCentros():
    def __init__(self, root_path):
        self.root_path = root_path
        self.csv_path = self.root_path + "classifications.csv"

    def pixels_adj(self, matriz, x, y, n_voltas):
        """
        Calcula as coordenadas dos pixels adjacentes a um pixel central em uma matriz, considerando um número de voltas (anéis) ao redor do pixel.

        Args:
            matriz (numpy.ndarray): A matriz que representa a imagem ou a área em questão.
            x (int): Coordenada x do pixel central.
            y (int): Coordenada y do pixel central.
            n_voltas (int): Número de voltas (ou anéis) a considerar ao redor do pixel central.

        Returns:
            list of tuple: Lista com as coordenadas dos pixels adjacentes (y, x).
        """

        coords_voltas = []
        n_linhas, n_colunas = matriz.shape

        for volta in range(1, n_voltas + 1):
            for i in range(-volta, volta + 1):
                if 0 <= y + i < n_linhas:
                    if 0 <= x - volta < n_colunas:
                        coords_voltas.append((y + i, x - volta))
                    if 0 <= x + volta < n_colunas:
                        coords_voltas.append((y + i, x + volta))

            for j in range(-volta + 1, volta):
                if 0 <= x + j < n_colunas:
                    if 0 <= y - volta < n_linhas:
                        coords_voltas.append((y - volta, x + j))
                    if 0 <= y + volta < n_linhas:
                        coords_voltas.append((y + volta, x + j))
        coords_voltas.append((y, x))
        return coords_voltas

    def verificar_adj(self, matriz, x, y, lista_centros):
        """
        Verifica se um pixel (x, y) esta adjacente a qualquer um dos centros de uma lista de centros.

        Args:
            matriz (numpy.ndarray): A matriz que representa a imagem ou a area em questão.
            x (int): Coordenada x do pixel a ser verificado.
            y (int): Coordenada y do pixel a ser verificado.
            lista_centros (list of tuple): Lista de coordenadas (y, x) de centros de pixels para comparacao.

        Returns:
            bool: Retorna True se o pixel (x, y) estiver adjacente a qualquer centro da lista, caso contrario, retorna False.
        """

        x, y = int(round(x)), int(round(y))
        if len(lista_centros) == 0:
            return False
        else:
            for each in lista_centros:
                each_x, each_y = int(round(each[1])), int(round(each[0]))
                ret = self.pixels_adj(
                    matriz=matriz, x=each_x, y=each_y, n_voltas=8)
                if (y, x) in ret:
                    return True
        return False

    def random_patchs_melhorados(self, patch_size, num_patches, imagem_tensor):
        """
        Gera coordenadas de centros de patches aleatórios em uma imagem, garantindo que os patches não se sobreponham de maneira extrema.

        Args:
            patch_size (tuple): Tamanho do patch, fornecido como (altura, largura).
            num_patches (int): Número total de patches a serem gerados.
            imagem_tensor (torch.Tensor): Tensor da imagem a ser processada, com dimensões (C, H, W).

        Returns:
            list of tuple: Lista de coordenadas (h, w) para os centros dos patches gerados.
        """

        img_PIL = Image.fromarray((imagem_tensor.permute(
            1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        image_height, image_width = img_PIL.size

        patch_height, patch_width = patch_size

        img_gray = img_PIL.convert('L')
        img_mtx = np.array(img_gray)

        centers = []

        for _ in range(num_patches):
            h = random.uniform(
                patch_height / 2, image_height - patch_height / 2)
            w = random.uniform(patch_width / 2, image_width - patch_width / 2)

            check = self.verificar_adj(img_mtx, w, h, centers)

            while check:
                h = random.uniform(
                    patch_height / 2, image_height - patch_height / 2)
                w = random.uniform(
                    patch_width / 2, image_width - patch_width / 2)
                check = self.verificar_adj(img_mtx, w, h, centers)

            centers.append((h, w))

        return centers

    def center_list_generate_randomico_aprimorado(self):
        """
        Gera uma lista de centros de patches aleatórios para todas as imagens de um diretório de dataset e os salva em um arquivo pickle.

        Args:
            Nenhum.

        Returns:
            None
        """

        diretorio_destino = f'{self.root_path}base_treinamento/'

        pickle_file = f'{self.root_path}/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl'

        img_size = (224, 224)
        batch_size = 32

        transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToTensor(),
        ])

        dataset = CustomImageFolder(
            root=diretorio_destino, transform=transform)

        print(f"Tamanho do dataset é de {len(dataset)} imagens")

        # lista_centro_dict = []

        centros_dict = {}

        # Iteração sobre as imagens do dataset
        # for img_idx in range(len(dataset)):
        for img_idx in tqdm(range(len(dataset)), desc="Processando imagens"):
            image, label, image_name = dataset[img_idx]

            centers = self.random_patchs_melhorados(patch_size=(
                16, 16), num_patches=196, imagem_tensor=image)

            centros_dict[image_name] = centers

        # Salvando o dicionário em um arquivo pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(centros_dict, f)

        print(f"Centros dos patches salvos em {pickle_file}")

    def generate_patch_centers(self, image_height, image_width, patch_size):
        """
        Realiza a extração dos patches de maneira padrão por Grid.

        Args:
            image_height (int): Altura da imagem.
            image_width (int): Largura da imagem.
            patch_size (int): Tamanho do patch (lado quadrado do patch).

        Returns:
            list of tuple: Lista de coordenadas (h, w) para os centros dos patches gerados.
        """

        stride = patch_size

        num_patches_h = image_height // stride
        num_patches_w = image_width // stride

        centers_h = []
        centers_w = []

        for i in range(num_patches_h):
            centers_h.append((i * stride + stride // 2))

        for j in range(num_patches_w):
            centers_w.append((j * stride + stride // 2))

        centers = []

        for h in centers_h:
            for w in centers_w:
                centers.append((h, w))

        return centers

    def remover_fundo_com_grabcut_recortado(self, imagem):
        """
        Remove o fundo de uma imagem usando o algoritmo GrabCut, recorta a área de interesse e retorna a imagem recortada, a máscara binária e a imagem original recortada.

        Args:
            imagem (numpy.ndarray): Imagem de entrada com fundo a ser removido.

        Returns:
            tuple: Contém três elementos:
                - Imagem recortada sem fundo (numpy.ndarray).
                - Máscara binária (numpy.ndarray).
                - Imagem original recortada (numpy.ndarray).
        """

        mascara = np.zeros(imagem.shape[:2], np.uint8)
        backgroundModel = np.zeros((1, 65), np.float64)
        foregroundModel = np.zeros((1, 65), np.float64)
        altura, largura = imagem.shape[:2]

        x1 = 0
        y1 = 0
        x2 = largura - 1
        y2 = altura - 1

        rectangle = (x1, y1, x2 - x1, y2 - y1)

        cv2.grabCut(imagem, mascara, rectangle,
                    backgroundModel, foregroundModel,
                    3, cv2.GC_INIT_WITH_RECT)

        mascara_objeto = np.where((mascara == 2) | (
            mascara == 0), 0, 1).astype('uint8')

        imagem_sem_fundo = imagem * mascara_objeto[:, :, np.newaxis]

        img_recortada = imagem_sem_fundo[y1:y2, x1:x2]

        imagem_gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

        _, mascara = cv2.threshold(imagem_gray, 10, 255, cv2.THRESH_BINARY)

        img_original_recortada = imagem[y1:y2, x1:x2]

        return img_recortada, mascara, img_original_recortada

    def is_image_black_percentage(self, image, threshold=0.9):
        """
        Verifica se uma imagem contém uma porcentagem de pixels pretos superior a um valor limite especificado.

        Args:
            image (numpy.ndarray): Imagem a ser analisada.
            threshold (float): Limite da porcentagem de pixels pretos para considerar a imagem como "preta" (default é 0.9).

        Returns:
            bool: Retorna True se a porcentagem de pixels pretos for maior ou igual ao limiar especificado, caso contrário, False.
        """

        total_pixels = image.size

        if image.ndim == 2:
            black_pixels = np.sum(image == 0)
        elif image.ndim == 3:
            black_pixels = np.sum(np.all(image == 0, axis=-1))

        black_percentage = black_pixels / total_pixels
        return black_percentage >= threshold

    def validar_centros(self, centros, patch_size, tamanho_img=(224, 224)):
        """
        Valida os centros dos patches, garantindo que eles nao ultrapassem os limites da imagem e que os patches caibam na imagem.

        Args:
            centros (list of tuple): Lista de coordenadas (h, w) dos centros dos patches.
            patch_size (int): Tamanho do patch (lado quadrado do patch).
            tamanho_img (tuple): Tamanho da imagem (altura, largura) para validação (default é (224, 224)).

        Returns:
            list of tuple: Lista de coordenadas (h, w) dos centros dos patches válidos.
        """

        offset = patch_size // 2
        centros_validos = []

        for (x, y) in centros:
            if (x - offset >= 0 and x + offset - 1 < tamanho_img[0]) and (y - offset >= 0 and y + offset - 1 < tamanho_img[1]):
                centros_validos.append((x, y))

        return centros_validos

    def grabcutextractcenters(self, imagem_tensor, tamanho_img=(224, 224), stride=16):
        """
        Extrai os centros de patches de uma imagem utilizando a segmentação GrabCut.

        Parâmetros:
        - imagem_tensor (torch.Tensor): A imagem de entrada no formato de tensor.
        - tamanho_img (tuple, opcional): O tamanho desejado para redimensionar a imagem. Default é (224, 224).
        - stride (int, opcional): O tamanho do stride para gerar patches. Default é 16.

        Retorna:
        - list: Uma lista contendo os centros de patches válidos extraídos da imagem.
        """

        imagem = imagem_tensor.permute(1, 2, 0).cpu().numpy()

        qtd_patches = int((tamanho_img[0]/stride) * (tamanho_img[0]/stride))

        # print(f'Quantidade de patches: {qtd_patches}')

        if imagem.max() <= 1:
            imagem = (imagem * 255).astype(np.uint8)

        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

        _, mask, _ = self.remover_fundo_com_grabcut_recortado(imagem=imagem)

        if self.is_image_black_percentage(mask):

            altura, largura, _ = imagem.shape

            centers_merge = self.generate_patch_centers(
                altura, largura, stride)

            return centers_merge
        else:
            mask = cv2.resize(mask, tamanho_img)

            centers_randomicos = []
            centers_randomicos_validos = []
            centers_stride = []
            centers_stride_validos = []
            centers_merge = []
            altura, largura = mask.shape

            for i in range(altura):
                for j in range(largura):
                    pixel = mask[i, j]

                    if len(pixel.shape) == 0:
                        if pixel == 255:
                            centers_randomicos.append((i, j))
                    else:
                        if np.array_equal(pixel, [255, 255, 255]):
                            centers_randomicos.append((i, j))

            centers_randomicos_validos = self.validar_centros(
                centros=centers_randomicos, patch_size=stride)

            for i in range(0, altura, stride):
                for j in range(0, largura, stride):
                    pixel = mask[i, j]

                    if len(pixel.shape) == 0:
                        if pixel == 255:
                            centers_stride.append((i, j))
                    else:
                        if np.array_equal(pixel, [255, 255, 255]):
                            centers_stride.append((i, j))

            centers_stride_validos = self.validar_centros(
                centros=centers_stride, patch_size=stride)

            quantidade_patches_stride = len(centers_stride_validos)

            if quantidade_patches_stride < qtd_patches:
                diferentes_lista1 = set(
                    centers_randomicos_validos) - set(centers_stride_validos)
                diferentes_lista2 = set(
                    centers_stride_validos) - set(centers_randomicos_validos)

                resultado = list(diferentes_lista1) + list(diferentes_lista2)

                random.shuffle(resultado)
                qtd_faltante_patches = qtd_patches - quantidade_patches_stride

                centers_merge = centers_stride_validos.copy()
                centers_merge.extend(resultado[0:qtd_faltante_patches])
            elif quantidade_patches_stride == qtd_patches:
                centers_merge = centers_stride_validos.copy()

            # fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # img_original_recortada = cv2.resize(imagem, tamanho_img)

            # axes[0].imshow(mask, cmap='gray')
            # axes[0].set_title('Máscara')
            # axes[0].axis('off')

            # axes[1].imshow(img_original_recortada)
            # axes[1].set_title('Imagem Original Recortada')
            # axes[1].axis('off')

            return centers_merge

    def center_list_generate_selecao_segmentacao(self):
        """
        Gera e salva a lista de centros de patches para todas as imagens de um dataset utilizando o GrabCut.

        Retorna:
        - None: O resultado é salvo em um arquivo pickle no diretório especificado.
        """

        diretorio_destino = f'{self.root_path}base_treinamento/'
        pickle_file = f'{self.root_path}/centros_pre_salvos/segmentacao_dicionario.pkl'

        img_size = (224, 224)
        batch_size = 32

        transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToTensor(),
        ])

        dataset = CustomImageFolder(
            root=diretorio_destino, transform=transform)

        centros_dict = {}

        # Iteração sobre as imagens do dataset
        for img_idx in tqdm(range(len(dataset)), desc="Processando imagens"):
            # Carrega a imagem e o rótulo
            image, label, image_name = dataset[img_idx]

            centers = self.grabcutextractcenters(
                image, stride=32)

            centros_dict[image_name] = centers

        with open(pickle_file, 'wb') as f:
            pickle.dump(centros_dict, f)

        print(f"Centros dos patches salvos em {pickle_file}")
