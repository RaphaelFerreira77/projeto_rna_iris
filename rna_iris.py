# =============================================================================
# REDE NEURAL ARTIFICIAL (4-8-3) DO ZERO - CLASSIFICAÇÃO DO DATASET IRIS
# Disciplina: Inteligência Artificial - UNIPAC Barbacena
# =============================================================================

# -----------------------------------------------------------------------------
# SEÇÃO 1: IMPORTS E SEMENTE ALEATÓRIA
# -----------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

SEED = 42                # semente para reproduzir sempre os mesmos resultados
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# SEÇÃO 2: CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# -----------------------------------------------------------------------------

def carregar_e_preparar_dados(seed=SEED):
    """
    Carrega o Iris e devolve os dados prontos para a rede neural:
    X_treino, X_teste (padronizados) e y_treino, y_teste (one-hot).
    Também devolve os rótulos originais do teste (úteis na avaliação).
    """
    # 1) Carregar o dataset: X tem shape (150, 4), y tem shape (150,)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2) Dividir em 80% treino / 20% teste (ANTES de padronizar)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.20,
        random_state=seed,
        stratify=y          # mantém a proporção das 3 classes
    )

    # 3) Padronização: média e desvio calculados SÓ com o treino
    media = X_treino.mean(axis=0)     # uma média por característica
    desvio = X_treino.std(axis=0)     # um desvio por característica
    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio   # usa média/desvio do TREINO

    # 4) One-Hot Encoding: transforma rótulo 0,1,2 em vetor de tamanho 3
    num_classes = 3
    y_treino_onehot = np.eye(num_classes)[y_treino]
    y_teste_onehot = np.eye(num_classes)[y_teste]

    return X_treino, X_teste, y_treino_onehot, y_teste_onehot, y_teste


# -----------------------------------------------------------------------------
# SEÇÃO 3: INICIALIZAÇÃO DOS PARÂMETROS DA REDE (ARQUITETURA 4-8-3)
# -----------------------------------------------------------------------------

def inicializar_parametros(n_entrada=4, n_escondida=8, n_saida=3, seed=SEED):
    """
    Cria os pesos (W1, W2) e os vieses (b1, b2) com valores aleatórios.

    Formas (shapes):
        W1: (n_entrada, n_escondida)  -> (4, 8)
        b1: (1, n_escondida)          -> (1, 8)
        W2: (n_escondida, n_saida)    -> (8, 3)
        b2: (1, n_saida)              -> (1, 3)
    """
    rng = np.random.default_rng(seed)

    # Pesos: números aleatórios de uma normal (média 0, desvio 1),
    # multiplicados por sqrt(2 / n_entradas_da_camada) (inicialização "He").
    W1 = rng.standard_normal((n_entrada, n_escondida)) * np.sqrt(2.0 / n_entrada)
    W2 = rng.standard_normal((n_escondida, n_saida)) * np.sqrt(2.0 / n_escondida)

    # Vieses: valores aleatórios bem pequenos
    b1 = rng.standard_normal((1, n_escondida)) * 0.01
    b2 = rng.standard_normal((1, n_saida)) * 0.01

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# -----------------------------------------------------------------------------
# TESTE DA ETAPA 2 (será reorganizado nas próximas etapas)
# O bloco abaixo precisa ficar SEMPRE no final do arquivo.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    X_treino, X_teste, y_treino, y_teste, y_teste_rotulos = carregar_e_preparar_dados()
    print("Shape X_treino:", X_treino.shape)
    print("Shape y_treino (one-hot):", y_treino.shape)

    parametros = inicializar_parametros()

    print("\n--- Parâmetros da rede 4-8-3 ---")
    for nome, valor in parametros.items():
        print(f"{nome}: shape = {valor.shape}")

    total = sum(valor.size for valor in parametros.values())
    print("\nTotal de parâmetros treináveis:", total)

    print("\nW1 (4x8), arredondado:")
    print(np.round(parametros["W1"], 3))
    print("\nb1 (1x8), arredondado:")
    print(np.round(parametros["b1"], 3))
    print("\nW2 (8x3), arredondado:")
    print(np.round(parametros["W2"], 3))
    print("\nb2 (1x3), arredondado:")
    print(np.round(parametros["b2"], 3))

    p2 = inicializar_parametros()
    print("\nReprodutível com a mesma semente?",
          all(np.array_equal(parametros[k], p2[k]) for k in parametros))