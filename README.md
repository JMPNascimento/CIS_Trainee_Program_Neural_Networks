# Star, Galaxy and Quasar Classification

Este projeto tem como objetivo classificar estrelas, galáxias e quasares com base em atributos numéricos, utilizando redes neurais profundas (MLP) e técnicas de ajuste de hiperparâmetros com Keras Tuner.

##  Dataset

O conjunto de dados utilizado foi `star_classification.csv`, contendo atributos numéricos de objetos celestes e uma coluna de classe (`class`) com a categoria da estrela.

## Etapas do Projeto

1. **Leitura e Análise dos Dados**  
   - Carregamento do dataset com `pandas`.
   - Verificação de tipos e valores ausentes.
   - Visualização inicial com `df.head()`.

2. **Visualização de Outliers**  
   - Seleção apenas dos atributos numéricos.
   - Geração de boxplots horizontais com `plotly.express` para análise visual dos outliers.

3. **Tratamento de Outliers**  
   - Implementação da função `detectar_outliers_iqr` baseada no método IQR.
   - Substituição de outliers pelos limites superiores/inferiores definidos pelo IQR.

4. **Preparo para o Modelo**  
   - Separação entre atributos (`X`) e classe (`y`).
   - Codificação da variável `y` com `LabelEncoder` e `to_categorical`.
   - Divisão dos dados em treino e teste.

5. **Treinamento da Rede MLP**  
   - Arquitetura:  
     - Entrada com `Input(shape)`.
     - Camadas densas com `relu` e `Dropout`.
     - Saída com `softmax`.
   - Compilação com otimizador Adam e `categorical_crossentropy`.
   - Treinamento com 50 épocas.

6. **Avaliação do Modelo**  
   - Avaliação nos dados de teste.
   - Geração de relatório de classificação.
   - Matriz de confusão com `seaborn`.

7. **Padronização dos Dados**  
   - Padronização com `StandardScaler` para aplicar o Keras Tuner.

8. **Ajuste de Hiperparâmetros com Keras Tuner**  
   - Hiperparâmetros ajustados:
     - Número de neurônios na camada oculta (`units`).
     - Taxa de dropout.
     - Taxa de aprendizado.
   - Uso do `RandomSearch` com parada antecipada (`EarlyStopping`).

9. **Avaliação do Melhor Modelo**  
   - Treinamento final com os melhores hiperparâmetros.
   - Avaliação final e nova matriz de confusão.

## Bibliotecas Utilizadas

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly.express`
- `scikit-learn`
- `tensorflow.keras`
- `keras_tuner`

## Requisitos

- Python 3.7+
- TensorFlow 2.x
- Keras Tuner
- Scikit-learn
- Plotly

## Resultados Esperados

- Acurácia de teste da MLP básica e do modelo ajustado.
- Relatório de classificação e matriz de confusão com desempenho por classe.

## Instalação

Você pode instalar todas as dependências com o seguinte comando:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn tensorflow keras-tuner

