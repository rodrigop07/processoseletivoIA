import os
import tf_keras as keras
import numpy as np

# carrega o dataset MNIST (dígitos escritos à mão 0-9)
# x_train: imagens de treino (60.000 amostras de 28x28 pixels)
# y_train: rótulos de treino (dígito correspondente a cada imagem)
# x_test: imagens de teste (10.000 amostras)
# y_test: rótulos de teste
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prepara os dados para a rede neural
# normaliza os valores dos pixels para o intervalo [0, 1] dividindo por 255
# expand_dims adiciona uma dimensão de canal (1 = escala de cinza)
# forma final dos dados: (número de amostras, 28, 28, 1)
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

# definição sa arquitetura da rede neural
# cria uma rede neural convolucional (CNN) sequencial para classificação de dígitos
model = keras.Sequential([
    # camada 1: Convolução com 16 filtros (3x3) + ReLU
    # extrai características iniciais da imagem
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # pooling: reduz dimensionalidade mantendo as características mais importantes
    keras.layers.MaxPooling2D((2, 2)),
    
    # camada 2: Convolução com 32 filtros (3x3) + ReLU
    # extrai características mais complexas
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # pooling: reduz novamente as dimensões
    keras.layers.MaxPooling2D((2, 2)),
    
    # flatten: converte matriz 2D em vetor 1D para as camadas densas
    keras.layers.Flatten(),
    # camada densa com 64 neurônios + ReLU para aprendizado de padrões
    keras.layers.Dense(64, activation='relu'),
    # camada de saída: 10 neurônios (um para cada dígito 0-9) + softmax para probabilidades
    keras.layers.Dense(10, activation='softmax')
])

# ocmpilação do modelo
# prepara o modelo para treinamento
# optimizer='adam': algoritmo adaptativo para otimização de pesos
# loss='sparse_categorical_crossentropy': função de perda para classificação multiclasse
# metrics=['accuracy']: métrica para monitorar desempenho
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# treina o modelo nos dados de treinamento
print("Iniciando o Treinamento Base...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# métricas de avaliação do modelo treinado
# testa o modelo nos dados de teste para verificar seu desempenho
# x_test/y_test: dados nunca vistos durante o treinamento
# verbose=0: não exibe informações detalhadas da avaliação
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia Base no Teste: {test_acc:.2%}")

# salva o modelo treinado em formato HDF5 (.h5)
# será utilizado posteriormente no processo de otimização
model.save('model.h5')
print("\nModelo base salvo com sucesso como 'model.h5', pronto para otimização")