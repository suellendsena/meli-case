Case Campanha de Marketing Mercado Pago

==================================================

Este projeto tem como objetivo avaliar uma campanha de marketing relacionada a um novo produto do Mercado Crédito. A solução abrange todo o processo, desde o pré-processamento dos dados, análise descritiva, criação e seleção de *features*, separação dos conjuntos para treinamento de modelos, avaliação de técnicas de desbalanceamento de dados, validação cruzada, até a construção dos artefatos necessários para um eventual *deploy* em produção.

O objetivo consiste em desenvolver um modelo capaz de classificar se um cliente, após ser contatado por meio da campanha de marketing direto, aceitará ou não a oferta do novo produto de Mercado Crédito. 

# Etapas de desenvolvimento

Para garantir o bom funcionamento do desenvolvimento, recomenda-se que instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

## 1. Análise exploratória de dados

A análise exploratória dos dados está disponível em [notebooks/01-EDA.ipynb](notebooks/01-EDA.ipynb).  
Inicialmente, foi realizada uma análise de volumetria e univariada da base *raw*, com foco na identificação de amostras duplicadas, variáveis pouco informativas e oportunidades de criação de novas *features*. Em seguida, uma análise bivariada com a *target* e investigação de possíveis casos de *data leakage* que poderiam comprometer a imparcialidade dos modelos. Essa etapa foi essencial para orientar a construção da base processada, utilizada nas fases seguintes.

## 2. Geração da base processada

A finalidade dessa etapa é a construção de uma base pronta para o início do desenvolvimento do modelo. Então, o processo resultante proporcionará uma base pronta para consumo.

```bash
python src/data/basic_process.py
```

A tabela resultante estará na pasta `data/processed`.

## 3. Geração da base interim

O processo de criação de features será realizado nessa etapa, produzindo uma base pronta para a modelagem em `data/interim`. Para tal, o script de apoio está em [src/utils/transformers.py](src/utils/transformers.py), este servirá como um pacote interno do projeto, onde há classes auxiliares para a construção da pipeline completa de modelagem.

```bash
python src/features/build_features.py
```

## 4. *Split* das bases de treino e teste

Para garantir a capacidade de generalização do modelo, é essencial realizar a separação aleatória dos dados. Essa divisão entre conjuntos de treino e teste permite avaliar o desempenho do modelo de forma mais robusta, especialmente durante o processo de validação cruzada.

```bash
python src/data/train_test_split.py
```

As bases estarão disponíveis na pasta `data/train_test`.

## 6. *Feature Selection*

No processo de seleção de variáveis, foi utilizado o algoritmo **Boruta**, uma técnica baseada em florestas aleatórias que identifica de forma robusta as *features* mais relevantes para o modelo. Ao finalizar a execução, será gerado um arquivo das features selecionadas e rejeitadas ([features_selected.yaml](src/features/selected/features_selected.yaml))

```bash
python src/features/feature_selection.py
```

## 7. *Encoders*

Os *encoders* serão gerados nessa etapa e disponibilizados na pasta `models`, assim como as bases *encodadas* para o início da modelagem. Esse procedimento é essencial para ter controle nos *inputs* dos modelos, como: preenchimentos de dados vazios, correção de *strings*, agrupamento de variáveis de alta cardinalidade e padronização de tipagem.

```bash
python src/features/create_encoder.py
```

## 8. *Model selection*

Os modelos *baseline* propostos nesta etapa foram: Decision Tree, Random Forest, Gradient Boosting Trees, AdaBoost, XGBoost e LightGBM. Essas escolhas foram feitas devido ao desempenho superior desses algoritmos em comparação com modelos paramétricos, além de sua maior flexibilidade. Esses modelos não exigem a normalização das features, nem demandam atenção especial à correlação entre as variáveis preditoras, entre outros pré-requisitos comuns em outras abordagens. Para a abordagem de classificação, foi utilizada a validação cruzada estratificada, considerando o desbalanceamento da variável target. Essa técnica garante que a proporção das classes seja preservada em cada divisão, proporcionando uma avaliação mais consistente e representativa do desempenho do modelo. Por fim, foi avaliado técnicas de balanceamento de dados como *Oversampling*, *Undersampling* e *SMOTE (Synthetic Minority Over-sampling Technique)*

## 9. *Tunning* de hiperparâmetros

O modelo *baseline* de melhor desempenho identificado na etapa anterior foi selecionado para o processo de *tuning*. Para essa tarefa, utilizou-se o **Optuna**, este módulo permite personalizar a função objetivo a ser otimizada. Diferentemente dos métodos tradicionais de busca de hiperparâmetros, o **Optuna** realiza uma exploração inteligente do espaço de hiperparâmetros, por meio de otimização bayesiana, ajustando-se com base nos resultados empíricos das iterações anteriores para melhorar a iteração do momento.

```bash
python src/models/tunning.py
```

## 10. *Model training*

O modelo final será obtido nessa etapa, em [notebooks/05-Model.ipynb](notebooks/05-Model.ipynb).  Os modelos serão salvos como um arquivo `.pkl` na pasta `models/predictors` para ser utilizado posteriormente na análise de resultados.

## 11. Geração dos artefatos

Esta etapa representa a fase final do desenvolvimento do modelo, onde será criada a pipeline definitiva para produção. Com isso, o modelo estará preparado para ser produtizado em ambientes em *real-time*, recebendo como entrada um JSON (payload) e retornando como saída um *score*.

```bash
python src/models/generate_artifacts.py
```

---

## 12. Análise dos resultados

Os resultados obtidos podem ser consultados em [notebooks/06-Resultados.ipynb](notebooks/06-Resultados.ipynb), bem como breves análises a respeito das métricas e interpretações.


------------

Organização do projeto


    ├── README.md                       <- README do projeto para guiar a sua execução.
    │
    ├── data
    │   ├── interim                     <- Dados intermediários, transformados a partir dos dados raw.
    │   ├── processed                   <- Dados finais: dados canônicos para o processo de modelagem.
    │   └── raw                         <- Dados raw, originais e imutáveis.
    │
    ├── notebooks                       <- Jupyter notebooks para análise descritiva, 
    │                                      acompanhamento do desenvolvimento e processos interativos.
    │
    ├── reports                         <- Relatórios do desenvolvimento.
    │   └── logs                        <- Logs dos scripts executáveis.
    │   └── visualization               <- Gráficos, imagens, evidências.
    │
    ├── requirements.txt                <- Lista de dependências do projeto. 
    │                                      Gerado com `pip freeze > requirements.txt`.
    │
    ├── pyproject.toml                  <- Torna o projeto instalável (`pip install -e .`). 
    │                                      Permite que `src` possa ser importado como módulo.
    │
    ├── src                             <- Código-fonte do projeto.
    │   │
    │   ├── __init__.py                 <- Torna `src` um módulo Python.
    │   │
    │   ├── data                        <- Scripts de processamento intermediário.
    │   │   └── basic_process.py
    │   │
    │   ├── features                    <- Scripts de criação e seleção de features.
    │   │   └── build_features.py
    │   │   └── create_encoders.py
    │   │   └── feature_selection.py
    │   │   └── selected                <- Contém os arquivos de features selecionadas.
    │   │
    │   ├── models                      <- Scripts relacionados à modelagem.
    │   │   └── generate_artifacts.py
    │   │   └── model_selection.py
    │   │   └── tunning.py
    │   │
    │   └── utils                       <- Funções auxiliares internas do projeto.
    │
    └── models                          <- Modelos treinados e serializados 
        └── encoders                    <- Encoders utilizados no pipeline.
        └── predictors                  <- Modelos preditivos salvos.
        └── wrapped                     <- Artefatos finais prontos para uso.


--------


<p><small>Projeto baseado em <a target="_blank" href=https://drivendata.github.io/cookiecutter-data-science/>cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>