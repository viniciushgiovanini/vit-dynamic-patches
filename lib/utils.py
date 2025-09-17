import pickle


def load_dict(path):
    with open(path, "rb") as f:
        lista_centro_dict = pickle.load(f)

    return lista_centro_dict


def strategy_centers_patch(pde):
    pde_value = None
    if pde == "ra":
        print("Abordagem selecionada: Randomica Aprimorado")
        pde_value = load_dict(
            "./data/centros_pre_salvos/randomico_melhorado_identificador_por_imgname.pkl"
        )
    elif pde == "ss":
        print("Abordagem selecionada: Seleção por Segmentação")
        pde_value = load_dict(
            "./data/centros_pre_salvos/segmentacao_dicionario.pkl"
        )
    elif pde == "grid":
        print("Abordagem selecionada: Grid")
        pde_value = []
    elif pde == "sr":
        print("Abordagem selecionada: Seleção Randomica")
        pde_value = []
    elif pde == "zigzag":
        pde_value = load_dict("./data/centros_pre_salvos/zigzag_centers.pkl")
        print("Abordagem selecionada: Seleção por ZigZag")
    elif pde == "espiral":
        print("Abordagem selecionada: Seleção por Espiral")
        pde_value = load_dict("./data/centros_pre_salvos/espiral_centers.pkl")
    elif pde == "espiral_sem_sobre":
        print("Abordagem selecionada: Seleção por Espiral - SEM SOBREPOSIÇÃO")
        pde_value = load_dict(
            "./data/centros_pre_salvos/espiral_sem_sobrepoisicao.pkl"
        )

    if pde_value == None and pde != "grid" and pde != "sr":
        raise Exception("PDE selecionada não existe !")

    return pde_value
