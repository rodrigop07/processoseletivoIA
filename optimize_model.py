import os
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import numpy as np

# carrega o modelo neural treinado anteriormente do arquivo 'model.h5'
print("Carregando o modelo base 'model.h5'...")
base_model = keras.models.load_model('model.h5')


# carrega os dados MNIST necessários para o fine-tuning
# o modelo precisa ser re-treinado após a otimização para recuperar a acurácia perdida
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# normaliza os pixels para [0, 1] e adiciona dimensão de canal
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)

# técnica 1: pruning (poda de pesos)
# remove pesos insignificantes (próximos a zero) da rede neural
# objetivo: reduzir o tamanho do modelo mantendo a acurácia
print("\nAplicando Pruning (poda de 50%)")
# função que aplica a técnica de poda seletiva ao modelo
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# configuração dos parâmetros de treinamento
# define hyperparâmetros para o fine-tuning após a poda
batch_size = 64  # quantidade de amostras por iteração
epochs = 2      # número reduzido de épocas (apenas 2) suficiente para recuperação
end_step = np.ceil(len(x_train) / batch_size).astype(np.int32) * epochs  # total de passos

# parâmetro de Poda
# configura o cronograma de poda (de 0% a 50% de esparsidade)
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0,        # começa sem poda
          final_sparsity=0.50,         # termina podando 50% dos pesos
          begin_step=0,                # início da poda
          end_step=end_step)           # fim da poda
}

# aplica a estratégia de poda configurada ao modelo base
model_for_pruning = prune_low_magnitude(base_model, **pruning_params)

# Recompila o modelo com os parâmetros de otimização
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# fine-tuning pós poda
# treina o modelo novamente para recuperar a acurácia após remover pesos
# callback UpdatePruningStep atualiza o cronograma de poda durante o treinamento
print("Realizando Fine-Tuning para recuperar acurácia...")
model_for_pruning.fit(
    x_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.1,  # 10% dos dados para validação
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]  # atualiza poda a cada step
)

# remoção das camadas de poda
# remove as camadas de suporte de poda para gerar um modelo limpo
# o resultado é um modelo mais compacto mas sem as estruturas de poda
model_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# técnica 2: dynamic range quantization
# reduz a precisão dos pesos (32-bit float → 8-bit int) para diminuir tamanho
# TFLite é o formato otimizado para dispositivos móveis e embarcados
print("\nAplicando Quantização e Convertendo para TFLite")
# cria conversor de modelo Keras para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model_export)
# ativa otimizações padrão (principalmente quantização)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# realiza a conversão final para o formato TFLite quantizado
tflite_quant_model = converter.convert()

# define o caminho do arquivo de saída
tflite_file_path = 'model.tflite'
# salva o modelo otimizado em arquivo binário
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_quant_model)

# exibe mensagem de sucesso
print(f"\nOtimização Dupla Concluída! Modelo final salvo como '{tflite_file_path}'")

# compara o tamanho original vs. otimizado para demonstrar ganho de compressão
tamanho_h5 = os.path.getsize('model.h5') / 1024
tamanho_tflite = os.path.getsize(tflite_file_path) / 1024
print(f"Tamanho (.h5 Base): {tamanho_h5:.2f} KB")
print(f"Tamanho (.tflite Pruning + Quantização): {tamanho_tflite:.2f} KB")