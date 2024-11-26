# Aplicação de Patches Dinâmicos em Vision Transformers em exames de Papanicolau

## Tempo de Desenvolvimento do Projeto
<a href="https://wakatime.com/badge/user/e5eaffb7-e096-4b18-a852-b6d7066b0044/project/75bd04cf-06b9-44bc-97b5-65029d479467"><img src="https://wakatime.com/badge/user/e5eaffb7-e096-4b18-a852-b6d7066b0044/project/75bd04cf-06b9-44bc-97b5-65029d479467.svg" alt="wakatime"></a>

## Explicações


- Este trabalho explora a aplicação de Vision Transformers (ViTs) na classificação de imagens de exames de Papanicolau, com o objetivo de identificar alterações citológicas. Foram realizados experimentos de *fine-tuning* com os modelos Tiny, Small e Base, sendo os modelos Small e Base utilizados como *baseline* para comparação. O foco da pesquisa está na modificação do módulo de extração de patches, que, no modelo padrão, utiliza projeção convolucional baseada em um grid fixo.

<br>
<center>
<figure>
<img src="doc/introducao2.drawio.png" alt="Arquitetura do ViT" width="600">
  <figcaption>Entrada e Saída do Vision Transformer</figcaption>
</figure>
</center>
<br>

- Neste projeto, propõe-se a substituição dessa abordagem tradicional por técnicas dinâmicas de extração de patches, combinando projeções lineares e convolucionais. Três métodos foram desenvolvidos e analisados: **Seleção Randômica (SR)**, **Randômica Aprimorada (RA)** e **Seleção por Segmentação (SS)**. Essas técnicas pretendem aprimorar a representatividade dos patches extraídos, explorando informações das imagens para melhorar a acurácia na classificação.







## Dataset e Pré-Processamentos

- Para gerar os patches recortados e balanceados, é necessário extrair a base de dados presente no site do [CRIC](https://database.cric.com.br/downloads), na qual tem a pasta chamada base dentro do diretório **data/base/**.
- Para realizar o recorte é necessário executar o notebook crop.ipynb, dentro do diretório **notebook/pre-processamento/crop.ipynb**. Logo após, será gerado 11.534 imagens recortadas em seus respectivos diretórios referentes às classes, presentes na pasta base_recortada. A próxima etapa é realizar a divisão dos dados em treino validação e teste, através do arquivo **notebook/pre-processamento/divisao.ipynb**, essa divisão será realizada dentro da pasta base_treinamento, e logo após para realizar o balanceamento de dados, é preciso rodar o arquivo **notebook/pre-processamento/balanceamento.ipynb**, na qual está projetando todas as classes para 1.000 imagens no treinamento, no próprio diretório base_treinamento, como a tabela abaixo.

<br>
<center>

| **Classe**   | **ASC-H** | **ASC-US** | **HSIL** | **LSIL** | **NFIL** | **SCC** |
|--------------|-----------|------------|----------|----------|----------|---------|
| **Train**    | 1.000     | 1.000      | 1.000    | 1.000    | 1.000    | 1.000   |
| **Validation** | 185       | 122        | 341      | 272      | 1.356    | 33      |
| **Test**     | 148       | 96         | 272      | 217      | 1.084    | 25      |

</center>


## Rodando o ViT

- O arquivo principal para treinamento do ViT padrão sem modificação é denominado **traning.py**, localizado na raiz do projeto, sendo necessário passar as pastas de teste e validação.

- Para rodar o ViT custom, com projeção linear ou convolucional deve rodar o arquivo denominado **traning_custom.py**, dentro desse arquivo está setado por padrão rodar com projeção linear, mas está comentado a parte do código necessário trocar para rodar a projeção convolucional. Esse método precisa do Pré-Processamento, que consiste na geração dos arquivos de centros de todos os patches.


## Fluxos de Teste

- Este projeto propõe os métodos **Seleção Randômica (SR)**, **Randômica Aprimorada (RA)** e **Seleção por Segmentação (SS)**, como mostrado na figura abaixo. Para utilizar o modelo custom de ViT, é necessário realizar a geração da lista de centros, que basicamente são os pixels centrais de cada patch, para a abordagem SS e a RA, na qual não é possível gerar esses pixels centrais durante o treinamento, dessa forma é necessário rodar o arquivo dentro do diretório **notebook/center_generation/generate_centers.ipynb**, e depois mover os arquivos .pkl para a pasta **data/centro_pre_salvos/**, e selecionar nas classes custom quais arquivos serão lidos no construtor da classe.

- O trabalho propõe a utilização dos modelos ViT-Base e ViT-Small, com projeção convolucional e extração por Grid, comparado com os mesmos modelos, através de projeção linear no modelo ViT-Base e no ViT-Small, e com projeção convolucional somente no ViT-Small, como ilustrado na figura abaixo.

<br>
<center>
<figure>
<img src="doc/fluxo_testes_teste.drawio.png" alt="Arquitetura do ViT" width="2600">
  <figcaption>Fluxos de Testes do Vision Transformer</figcaption>
</figure>
</center>
<br>



## Instalando venv

- Crie a virtual environment no dirétorio raiz.

```shelll
python -m venv venv
```

- Entre na venv (Windows Powershell)

```shell
cd venv/Scripts/
./activate
```

- Com ela ativada instale as dependências

```shell
pip install -r requirements.txt
```

- Verifique a instalação

```shell
pip list
```  
## Dirétorios 

- Todos os notebooks estão lendo dados da pasta **data**.

```md
.
├── README.md
├── classes
│   ├── CustomImageFolder.py
│   ├── Validate.py
│   ├── dynamic_patches.py
│   ├── modelo.py
│   ├── modelo_binario.py
│   ├── modelo_custom.py
│   ├── modelo_custom_conv2d.py
│   └── patch_visualizer.py
├── data
│   ├── base
│   ├── base_mascara
│   ├── base_recortada
│   ├── base_save
│   ├── base_temporaria
│   ├── base_treinamento
│   ├── centros_pre_salvos
│   ├── classifications.csv
├── grafico-animado.py
├── graph
├── lightning_logs
├── models
├── notebook
│   ├── Heatmap
│   ├── Mask
│   ├── conv2d.py
│   ├── doc
│   ├── linear.py
│   ├── patch.ipynb
│   ├── pre-processamento
│   └── tratamento.ipynb
├── requirements.txt
├── avaliar-modelo.py
├── traning.py
└── validacao.ipynb
```

- O path **base** contém o dataset original, e o **base_temporaria** é uma copia, para manter os dados.
- O path **base_recortada**, possui subpastas com cada tipo de célula e com as celulas recortada de um tamanho 100 para os lados por padrão.

## Verificação do CUDA

- Verifique a versão do Cuda instalada por default no WSL2
```
nvidia-smi
watch -n 1 nvidia-smi
```


## Linguagens de Desenvolvimento

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytorch/pytorch-original.svg" width="50px"/>

## Desenvolvimento ✏

**Feito por**: [Vinícius Henrique Giovanini](https://github.com/viniciushgiovanini)