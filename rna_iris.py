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
# TESTE DA ETAPA 1 (será removido/reorganizado nas próximas etapas)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    X_treino, X_teste, y_treino, y_teste, y_teste_rotulos = carregar_e_preparar_dados()

    print("Shape X_treino:", X_treino.shape)
    print("Shape X_teste :", X_teste.shape)
    print("Shape y_treino (one-hot):", y_treino.shape)
    print("Shape y_teste  (one-hot):", y_teste.shape)

    print("\nMédia das características no treino (esperado ~0):")
    print(np.round(X_treino.mean(axis=0), 4))
    print("Desvio das características no treino (esperado ~1):")
    print(np.round(X_treino.std(axis=0), 4))

    print("\nPrimeiras 3 amostras de treino (padronizadas):")
    print(np.round(X_treino[:3], 3))
    print("Seus rótulos one-hot:")
    print(y_treino[:3])
