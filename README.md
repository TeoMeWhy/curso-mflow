# Curso de MLFlow

Curso destinado a explicar o funcionamento e principais conceitos do MLFlow para desenvolvimento de modelos de Machine Learning e Inteligência Artificial.

O uso dessa ferramenta viabilisará a produção e gestão de diferentes modelos treinados. Bem como auxiliará na maneira como os modelos são selecionados para produção.

Todo material está no YouTube: [clique aqui](https://youtube.com/playlist?list=PLvlkVRRKOYFQeQEA5Lc0US9i-EK8eGgrs&si=_xiG7YzwzeVVra9t).

## Estrutura do projeto

```bash
.
├── mlflow-server
│   ├── mlartifacts
│   ├── mlruns
│   └── requirements.txt
├── model_churn
│   ├── data
│   ├── requirements.txt
│   └── train.py
└── README.md
```

### mlflow-server

É aqui que os artefatos serão gerados a partir do MLFlow. É o lugar onde você deve executar os comandos:

```bash
cd mlflow-server
conda create --name mlflow-server python=3.12
conda activate mlflow-server
pip install -r requirements.txt
mlflow server
```

Sempre que precisar subir o servidor do MLFlow, pode executar:
ml-churn
```bash
cd mlflow-server
conda activate mlflow-server
mlflow server
```

Assim que você criar os primeiros modelos e exeprimentos, as demais pastas serão criadas.

### model_churn

É o diretório do nosso projeto de Machine Learning. É onde contruímos o nosso modelo, armazenamos os dados, e todo código necessário para o modelo ser treinado e executado.

Para construir o ambiente necessário, execute os comandos:


```bash
cd model_churn
conda create --name ml-churn python=3.12
conda activate ml-churn
pip install -r requirements.txt
```