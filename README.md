# Dynamic Patches at Vision Transformers

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
├── README.md
├── avaliar-modelo.py
├── data
│   ├── base
│   ├── base_balanceada
│   ├── base_mascara
│   ├── base_mascara_70_value_expand.zip
│   ├── base_mascara_90_value_expand.zip
│   ├── base_recortada
│   ├── base_temporaria
│   ├── base_treinamento
│   ├── classifications.csv
│   └── classifications.txt
├── env
├── grafico-animado.py
├── graph
├── modelo.py
├── models
├── notebook
│   ├── balanceamento.ipynb
│   ├── crop.ipynb
│   ├── divisao.ipynb
│   ├── heatmap.ipynb
│   ├── mask.ipynb
│   ├── mask_genertion.ipynb
│   └── test
├── requirements.txt
├── tranining_multiclass.py
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