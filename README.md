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
data/
│
├── base/
│   └── img.png
│
├── base_recortada/
│    │
│    ├── LSIL
│    │   └── img.png
│    │
│    ├── Negative for intraepithelial lesion
│        └── img.png
│
└── base_temporaria/
    └── img.png
```

- O path **base** contém o dataset original, e o **base_temporaria** é uma copia, para manter os dados.
- O path **base_recortada**, possui subpastas com cada tipo de célula e com as celulas recortada de um tamanho 100 para os lados por padrão.

## Linguagens de Desenvolvimento

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytorch/pytorch-original.svg" width="50px"/>

## Desenvolvimento ✏

**Feito por**: [Vinícius Henrique Giovanini](https://github.com/viniciushgiovanini)