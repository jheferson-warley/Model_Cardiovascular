
# 📊 Previsão de Doenças Cardíacas com Random Forest

![Banner](https://github.com/user-attachments/assets/3fc8d8fb-035a-42a6-b8f5-2cbc9a76d877) 

---

## 📋 Visão Geral

Este projeto utiliza o algoritmo **Random Forest** para prever a presença de **doenças cardíacas** com base em um dataset de fatores de risco cardiovasculares. O objetivo é construir um modelo de machine learning que possa identificar pacientes com risco de doença cardíaca, utilizando variáveis como idade, hábitos alimentares, histórico de tabagismo e outras condições de saúde. Além disso, o projeto explora os dados com visualizações detalhadas e avalia o desempenho do modelo com métricas robustas, como ROC AUC e curva ROC.

O projeto foi desenvolvido como parte de um portfólio de ciência de dados, demonstrando habilidades em:
- **Pré-processamento de dados** (tratamento de valores nulos, codificação de variáveis categóricas, normalização).
- **Análise exploratória de dados (EDA)** com gráficos interativos.
- **Modelagem preditiva** com Random Forest.
- **Avaliação de modelos** usando métricas como acurácia, precisão, recall, F1-score e ROC AUC.
- **Visualização de resultados** para insights acionáveis.

---

## 🎯 Objetivo

Desenvolver um modelo de machine learning para prever a presença de doenças cardíacas (`Heart_Disease`) com base em variáveis como:
- **Demográficas**: Idade (`Age_Category`), sexo (`Sex`).
- **Saúde Geral**: Saúde autoavaliada (`General_Health`), IMC (`BMI`), histórico de tabagismo (`Smoking_History`).
- **Hábitos Alimentares**: Consumo de frutas (`Fruit_Consumption`), vegetais (`Green_Vegetables_Consumption`), batatas fritas (`FriedPotato_Consumption`).
- **Outras Condições**: Diabetes, artrite, depressão, entre outras.

O modelo foi avaliado com foco em sua capacidade de identificar casos positivos (doença cardíaca) em um dataset desbalanceado, usando métricas como ROC AUC e curvas ROC para medir a discriminação entre classes.

---

## 📊 Resultados Principais

- **Acurácia do Modelo**: 72.69% (teste), 73.07% (treino) → Indica ausência de overfitting significativo.
- **Relatório de Classificação**:
  - Classe "Sem Doença" (0): Precisão 98%, Recall 72%, F1-Score 83%.
  - Classe "Com Doença" (1): Precisão 20%, Recall 79%, F1-Score 32%.
  - **Insight**: O modelo é bom em capturar casos de doença (recall 79%), mas sofre com muitos falsos positivos (precisão 20%) devido ao desbalanceamento (92% "Sem Doença" vs. 8% "Com Doença").
- **ROC AUC**: 0.8294 → Boa capacidade de discriminação entre as classes.
- **Curva ROC**:

  ![output](https://github.com/user-attachments/assets/8ea8f9b1-82b7-40cc-9f4a-30c70ecbffd8)

  *A curva ROC mostra um bom desempenho inicial (alta TPR), mas a presença de falsos positivos impede um AUC mais próximo de 1.*

---

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal.
- **Pandas**: Manipulação e pré-processamento de dados.
- **Scikit-learn**: Modelagem (Random Forest), pré-processamento (StandardScaler, LabelEncoder), avaliação (ROC AUC, métricas de classificação).
- **Matplotlib/Seaborn**: Visualizações (gráficos de distribuição, curvas ROC, matriz de confusão).
- **IPython**: Formatação de saída com Markdown para apresentações.

---

## 📂 Estrutura do Projeto

```
Cardiovascular-Disease-Risk-Prediction/
│
├── data/
│   └── Cardiovascular-Diseases-Risk.csv  # Dataset utilizado
├── notebooks/
│   └── Cardiovascular_Risk_Analysis.ipynb  # Jupyter Notebook com o código completo
├── images/
│   └── curva_roc.png  # Gráficos gerados (ex.: Curva ROC)
├── README.md  # Este arquivo
└── requirements.txt  # Dependências do projeto
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
- Python 3.8 ou superior
- Git (para clonar o repositório)
- Ambiente virtual (recomendado)

### 2. Clonar o Repositório
```bash
git clone https://github.com/SEU_USUARIO/Cardiovascular-Disease-Risk-Prediction.git
cd Cardiovascular-Disease-Risk-Prediction
```

### 3. Criar e Ativar um Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 4. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 5. Executar o Notebook
- Abra o Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
- Navegue até `notebooks/Cardiovascular_Risk_Analysis.ipynb` e execute as células.

### 6. Dataset
- O arquivo `Cardiovascular-Diseases-Risk.csv` está na pasta `data/`. Certifique-se de que ele está no caminho correto ou ajuste o código conforme necessário.

---

## 📈 Metodologia

### 1️⃣ Carregamento e Pré-processamento
- **Dataset**: Contém variáveis como idade, sexo, IMC, hábitos alimentares e condições de saúde.
- **Pré-processamento**:
  - Tratamento de valores nulos (remoção de linhas com valores ausentes).
  - Codificação de variáveis categóricas usando `LabelEncoder`.
  - Normalização de variáveis numéricas com `StandardScaler`.

### 2️⃣ Análise Exploratória de Dados (EDA)
- **Distribuição de Doenças Cardíacas**: Análise do desbalanceamento entre as classes.
- **Idade e Doença Cardíaca**: Visualização da relação entre faixa etária e presença de doença.
- **Hábitos Alimentares**: Impacto do consumo de vegetais e frituras no risco cardíaco.

### 3️⃣ Modelagem com Random Forest
- **Divisão dos Dados**: 80% treino, 20% teste.
- **Modelo**: Random Forest com 100 árvores.
- **Ajustes**: Hiperparâmetros ajustados para evitar overfitting (ex.: `max_depth`, `min_samples_split`).

### 4️⃣ Avaliação do Modelo
- **Métricas**:
  - Acurácia, precisão, recall, F1-score.
  - ROC AUC (0.8294) para medir discriminação global.
- **Visualizações**:
  - Matriz de Confusão.
  - Curva ROC para avaliar trade-off entre TPR e FPR.
  - Importância das features para identificar variáveis mais influentes.

### 5️⃣ Insights e Visualizações
- **Fatores de Risco**: Variáveis como `Age_Category`, `BMI` e `Smoking_History` são as mais influentes.
- **Recomendações**: Foco em prevenção para grupos etários mais velhos e incentivo a dietas ricas em vegetais.

---

## 📉 Resultados Detalhados

### 1. Análise Exploratória
- **Distribuição de Doenças Cardíacas**:
  - 92% dos pacientes não têm doença cardíaca (classe 0).
  - 8% têm doença cardíaca (classe 1).
  - Insight: O desbalanceamento exige técnicas como SMOTE ou ajuste de pesos para melhorar a previsão da classe minoritária.

- **Idade e Doença Cardíaca**:
  - Pacientes em faixas etárias mais altas (ex.: 70+) têm maior prevalência de doenças cardíacas.
  - Insight: Campanhas de prevenção devem focar em idosos.

- **Hábitos Alimentares**:
  - Alto consumo de frituras está associado a maior risco de doença cardíaca.
  - Maior consumo de vegetais está associado a menor risco.
  - Insight: Mudanças na dieta podem ser uma estratégia de prevenção.

### 2. Desempenho do Modelo
- **Acurácia**: 72.69% (teste), mas influenciada pelo desbalanceamento.
- **Classe "Com Doença"**:
  - Recall de 79%: O modelo captura a maioria dos casos de doença.
  - Precisão de 20%: Muitos falsos positivos, indicando necessidade de ajuste.
- **ROC AUC**: 0.8294 → Boa discriminação, mas há espaço para melhoria na precisão.

### 3. Curva ROC 
A curva ROC mostra uma boa capacidade de discriminação (AUC = 0.8294), com alta TPR inicial, mas limitada por falsos positivos.

---

## 💡 Insights e Conclusões

- **Fatores de Risco**:
  - Idade avançada, tabagismo e alto IMC são os principais preditores de doenças cardíacas.
  - Dietas ricas em frituras aumentam o risco, enquanto o consumo de vegetais o reduz.

- **Desempenho do Modelo**:
  - O modelo é eficaz em identificar casos de doença cardíaca (recall 79%), mas a baixa precisão (20%) indica muitos falsos positivos.
  - O ROC AUC de 0.8294 confirma uma boa discriminação, mas ajustes podem melhorar a precisão.

- **Aplicações Práticas**:
  - Útil para triagem inicial em sistemas de saúde, identificando pacientes que precisam de exames adicionais.
  - Recomendações de saúde pública: Foco em prevenção para idosos e promoção de dietas saudáveis.

---

## 🚀 Possíveis Melhorias

- **Balanceamento das Classes**:
  - Aplicar SMOTE para gerar amostras sintéticas da classe minoritária.
  - Ajustar o limiar de decisão para melhorar a precisão da classe "Com Doença".

- **Otimização do Modelo**:
  - Usar `GridSearchCV` para encontrar os melhores hiperparâmetros.
  - Testar outros algoritmos, como XGBoost ou LightGBM, que podem lidar melhor com desbalanceamento.

- **Novas Features**:
  - Criar features derivadas (ex.: razão entre consumo de vegetais e frituras).
  - Incorporar dados adicionais, como pressão arterial ou níveis de colesterol.

---

## 📚 Referências

- Dataset: [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset) 
- Documentação do Scikit-learn: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- Documentação do Seaborn: [Visualization with Seaborn](https://seaborn.pydata.org/).

---

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tem sugestões para melhorar o modelo, adicionar novas análises ou otimizar o código, sinta-se à vontade para abrir uma issue ou enviar um pull request.

---

## 📧 Contato

- **Autor**: Jheferson Warley
- **GitHub**: [github.com/jheferson-warley](https://github.com/jheferson-warley)
- **Email**: [jhefersonwarley@gmail.com](mailto:jhefersonwarley@gmail.com)

---

⭐ **Se achou este projeto útil, deixe uma estrela no repositório!**

---
