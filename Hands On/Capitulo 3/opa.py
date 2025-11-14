# 1. Importações necessárias
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 2. Carregar os dados
# O 'parser="auto"' pode ser necessário em versões mais recentes do scikit-learn
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist["data"], mnist["target"]

# 3. Preparar os dados e dividir em treino/teste
# Conversão do tipo de 'y' para numérico
y = y.astype(np.uint8)

# Divisão padrão do MNIST (primeiros 60k para treino, últimos 10k para teste)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print("Dados divididos:")
print("Formato de X_train:", X_train.shape)
print("Formato de X_test:", X_test.shape)
print("-" * 30)


# --- CORREÇÃO: Pré-processamento Correto ---

# 4. Criar UMA instância do scaler
scaler = StandardScaler()

# 5. Ajustar o scaler APENAS com os dados de treino
print("Ajustando o scaler (fit) nos dados de treino...")
scaler.fit(X_train)

# 6. Transformar AMBOS os conjuntos com o MESMO scaler
print("Transformando os dados de treino e teste...")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Dados escalados com sucesso.")
print("-" * 30)


# 7. Treinar o modelo
print("Treinando o modelo KNeighborsClassifier...")
knn_model = KNeighborsClassifier(n_neighbors=10, weights="distance", leaf_size=5)
knn_model.fit(X_train_scaled, y_train)
print("Modelo treinado.")
print("-" * 30)


# --- CORREÇÃO: Previsões com Dados Corretamente Escalados ---

# 8. Fazer previsões
print("Realizando previsões...")
y_pred_train = knn_model.predict(X_train_scaled)
y_pred_test = knn_model.predict(X_test_scaled)


# 9. Avaliar os resultados
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)

print("\n--- RESULTADOS ---")
print(f"Acurácia no Conjunto de TESTE:  {accuracy_test:.4f}")
print(f"Acurácia no Conjunto de TREINO: {accuracy_train:.4f}")