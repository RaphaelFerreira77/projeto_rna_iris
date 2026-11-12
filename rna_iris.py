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
# SEÇÃO 4: FUNÇÕES DE ATIVAÇÃO, FORWARD PASS E FUNÇÃO DE PERDA
# -----------------------------------------------------------------------------

def relu(Z):
    """ReLU: devolve Z onde Z > 0 e 0 onde Z <= 0."""
    return np.maximum(0, Z)


def softmax(Z):
    """
    Softmax linha a linha: transforma cada linha de Z em probabilidades
    (todas >= 0 e somando 1).
    Subtraímos o maior valor de cada linha antes do exp() para evitar overflow
    (isso não muda o resultado final).
    """
    Z_estavel = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_estavel)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward(X, parametros):
    """
    Forward Pass: leva os dados da entrada até a saída da rede.

    X: (n_amostras, 4)
    Retorna:
        A2: (n_amostras, 3) probabilidades previstas
        cache: valores intermediários, que o Backpropagation vai precisar
    """
    W1, b1 = parametros["W1"], parametros["b1"]
    W2, b2 = parametros["W2"], parametros["b2"]

    Z1 = X @ W1 + b1        # (n, 4) @ (4, 8) + (1, 8) -> (n, 8)
    A1 = relu(Z1)           # (n, 8)
    Z2 = A1 @ W2 + b2       # (n, 8) @ (8, 3) + (1, 3) -> (n, 3)
    A2 = softmax(Z2)        # (n, 3)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def entropia_cruzada(Y, A2, eps=1e-12):
    """
    Cross-Entropy Loss média:  L = -(1/n) * soma( Y * log(A2) )

    Y : (n, 3) rótulos reais em one-hot
    A2: (n, 3) probabilidades previstas
    eps evita log(0), que seria -infinito.
    """
    A2 = np.clip(A2, eps, 1.0)
    perda_por_amostra = -np.sum(Y * np.log(A2), axis=1)   # (n,)
    return np.mean(perda_por_amostra)

# -----------------------------------------------------------------------------
# SEÇÃO 5: BACKPROPAGATION E GRADIENTE DESCENDENTE
# -----------------------------------------------------------------------------

def backward(X, Y, parametros, cache):
    """
    Backpropagation: calcula o gradiente da perda em relação a W1, b1, W2, b2
    usando a regra da cadeia, da saída para a entrada.

    X: (n, 4)   Y: (n, 3) one-hot
    Retorna um dicionário com dW1, db1, dW2, db2 (mesmos shapes dos parâmetros).
    """
    n = X.shape[0]
    W2 = parametros["W2"]
    A1, Z1, A2 = cache["A1"], cache["Z1"], cache["A2"]

    # --- Camada de saída ---
    # Softmax + Cross-Entropy juntas resultam em uma fórmula simples:
    # dL/dZ2 = (A2 - Y) / n
    dZ2 = (A2 - Y) / n                           # (n, 3)
    dW2 = A1.T @ dZ2                             # (8, n) @ (n, 3) -> (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True)     # (1, 3)

    # --- Camada escondida ---
    dA1 = dZ2 @ W2.T                             # (n, 3) @ (3, 8) -> (n, 8)
    dZ1 = dA1 * (Z1 > 0)                         # derivada da ReLU: 1 se Z1>0, senão 0
    dW1 = X.T @ dZ1                              # (4, n) @ (n, 8) -> (4, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True)     # (1, 8)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def atualizar_parametros(parametros, gradientes, taxa_aprendizado):
    """
    Gradiente Descendente: dá um passo contra o gradiente.
        parametro = parametro - taxa_aprendizado * gradiente
    """
    parametros["W1"] -= taxa_aprendizado * gradientes["dW1"]
    parametros["b1"] -= taxa_aprendizado * gradientes["db1"]
    parametros["W2"] -= taxa_aprendizado * gradientes["dW2"]
    parametros["b2"] -= taxa_aprendizado * gradientes["db2"]
    return parametros

# -----------------------------------------------------------------------------
# SEÇÃO 6: LOOP DE TREINAMENTO E GRÁFICO DA PERDA
# -----------------------------------------------------------------------------

def treinar(X, Y, parametros, epocas=1000, taxa_aprendizado=0.1, intervalo_log=100):
    """
    Treina a rede com Gradiente Descendente.
    Cada época = 1 passada completa pelos dados de treino:
        forward -> perda -> backward -> atualização dos parâmetros.

    Retorna os parâmetros treinados e a lista com a perda de cada época.
    """
    historico_perda = []

    for epoca in range(1, epocas + 1):
        # 1) Forward Pass
        A2, cache = forward(X, parametros)

        # 2) Perda (Cross-Entropy)
        perda = entropia_cruzada(Y, A2)
        historico_perda.append(perda)

        # 3) Backpropagation
        gradientes = backward(X, Y, parametros, cache)

        # 4) Atualização dos parâmetros (Gradiente Descendente)
        parametros = atualizar_parametros(parametros, gradientes, taxa_aprendizado)

        # Acompanhamento no console
        if epoca == 1 or epoca % intervalo_log == 0:
            print(f"Época {epoca:4d}/{epocas} | perda = {perda:.4f}")

    return parametros, historico_perda


def plotar_perda(historico_perda, caminho_arquivo="perda.png"):
    """Gera o gráfico da perda por época, salva em arquivo e exibe na tela."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historico_perda) + 1), historico_perda, color="tab:blue")
    plt.title("Evolução da Cross-Entropy Loss durante o treinamento")
    plt.xlabel("Época")
    plt.ylabel("Perda (Cross-Entropy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_arquivo, dpi=150)
    plt.show()

# -----------------------------------------------------------------------------
# EXECUÇÃO: TREINAMENTO (ETAPA 3 - PARTE C)
# O bloco abaixo precisa ficar SEMPRE no final do arquivo.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    EPOCAS = 1000
    TAXA_APRENDIZADO = 0.1

    # Dados: o conjunto de teste NÃO é usado no treinamento
    X_treino, X_teste, y_treino, y_teste, y_teste_rotulos = carregar_e_preparar_dados()

    # Parâmetros iniciais aleatórios
    parametros = inicializar_parametros()

    # Treinamento só com os dados de treino
    parametros, historico_perda = treinar(
        X_treino, y_treino, parametros,
        epocas=EPOCAS, taxa_aprendizado=TAXA_APRENDIZADO
    )

    print("\nPerda inicial:", round(historico_perda[0], 4))
    print("Perda final  :", round(historico_perda[-1], 4))

    plotar_perda(historico_perda)