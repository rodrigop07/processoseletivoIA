👤 Identificação: **Rodrigo Pinheiro Alcantara**

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura base foi definida no arquivo `train_model.py` utilizando uma **Rede Neural Convolucional (CNN)** focada em extração de características com baixo custo computacional.

O modelo sequencial é composto por:
* **Camadas de Extração:** Duas camadas convolucionais (`Conv2D`) de 16 e 32 filtros com kernel 3x3 (ativação ReLU), seguidas por camadas de subamostragem (`MaxPooling2D` 2x2) para redução de dimensionalidade.
* **Camada de Classificação:** Uma camada densa de 64 neurônios atuando sobre o mapa de características achatado (`Flatten`), finalizando com uma camada de saída com ativação **Softmax** para as 10 classes do dataset MNIST.

Para garantir modularidade e boas práticas de engenharia de software, o script de treinamento foi isolado apenas para a geração do modelo base, delegando o pipeline de redução para o script de otimização.

### 2️⃣ Bibliotecas Utilizadas

* **TensorFlow:** Motor principal para a construção e inferência.
* **tf_keras:** Utilizada para suportar as ferramentas de poda de pesos nas versões mais recentes do TensorFlow.
* **tensorflow-model-optimization (tfmot):** Biblioteca oficial do TF para aplicação de técnicas avançadas de TinyML/Edge AI.
* **NumPy:** Essencial para o pré-processamento e adequação dos tensores de imagem (28x28x1).

### 3️⃣ Técnica de Otimização do Modelo

No arquivo `optimize_model.py`, foi implementado um pipeline de **Otimização Dupla**, combinando duas das principais técnicas da indústria para Edge AI:

1. **Weight Pruning (Poda de Pesos):** Aplicada a técnica de `prune_low_magnitude`. O modelo base foi carregado e submetido a um *Fine-Tuning* de 2 épocas onde **50% das conexões menos relevantes foram zeradas** (esparsidade de 0.50). Isso reduz drasticamente a necessidade de processamento durante a inferência, já que o hardware pode ignorar cálculos multiplicados por zero.
2. **Dynamic Range Quantization:** Após a poda, o modelo "esparso" foi convertido via `TFLiteConverter`, esmagando os pesos restantes de ponto flutuante de 32 bits (`float32`) para inteiros de 8 bits (`int8`). 

Esta combinação garante um modelo final extremamente leve e rápido em arquiteturas de microcontroladores.

### 4️⃣ Resultados Obtidos

Para uma avaliação mais realista em cenários de classificação multiclasse, foi adicionada a métrica de **Top-3 Accuracy**, além da acurácia padrão.

* **Acurácia (Top-1):** 99.05%
* **Top-3 Accuracy:** 98.39% - *Demonstra que a resposta correta está quase sempre entre as 3 principais previsões da rede.*
* **Redução de Tamanho:** O arquivo final `.tflite` reduziu para 62.73 KB, enquanto o arquivo `h5` base obteve 707.55 KB, uma redução de quase 11x, combinando o formato binário reduzido com a matriz esparsa gerada pelo Pruning.

### 5️⃣ Comentários Adicionais (Opcional)

* **Decisões Técnicas:** A separação entre o treinamento base (`train_model.py`) e o *Fine-Tuning* com otimização (`optimize_model.py`) foi intencional. Isso evita que a rede perca drasticamente sua inteligência ao ter 50% de seus pesos cortados de uma só vez, permitindo que ela se reajuste às perdas durante as épocas extras.
* **Dificuldades:** A gestão de versões do ambiente foi resolvida através do uso de Containers (Docker), garantindo que as dependências do TensorFlow estivessem isoladas da versão do sistema anfitrião.
* **Aprendizados:** O desafio provou na prática como múltiplas métricas de Edge AI (Pruning e Quantização) podem ser empilhadas sem comprometer a acurácia, desde que haja um pipeline bem estruturado e focado na estabilidade numérica.