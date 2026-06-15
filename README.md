# CartPole Inverted Pendulum - Q-Learning

Este projeto implementa o clássico problema de controle do Pêndulo Invertido (*CartPole*) utilizando o algoritmo de Aprendizado por Reforço **Q-Learning** com discretização não linear de estados. A física do ambiente foi desenvolvida do zero em Python utilizando o integrador numérico **Runge-Kutta de 4ª Ordem (RK4)**.

## Como Rodar o Projeto

### 1. Instalar as Dependências

Antes de começar, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 2. Treinar o Agente

Para treinar o agente utilizando Q-Learning e salvar a tabela Q aprendida:

```bash
python train.py
```

Ao final do treinamento, será gerado o arquivo `q_table.npy`, contendo a política aprendida.

### 3. Executar o Agente Treinado

Para carregar a tabela Q treinada e visualizar o agente controlando o sistema:

```bash
python enjoy.py
```

## Scripts Adicionais

O projeto também inclui scripts auxiliares para experimentação e análise:

- `hyperparameter_study.py` — estudo e comparação de hiperparâmetros do treinamento.
- `play_manual.py` — controle manual do carrinho para demonstrações e testes.
- `train.py` — treinamento do agente Q-Learning.
- `enjoy.py` — execução da política treinada.

## Tecnologias Utilizadas

- Python
- NumPy
- Pygame
- Matplotlib

## Objetivo

O objetivo deste projeto é demonstrar, de forma didática, a construção completa de um ambiente CartPole sem o uso de bibliotecas prontas como Gymnasium ou Stable-Baselines, abordando desde a modelagem física até o treinamento do agente por Aprendizado por Reforço.