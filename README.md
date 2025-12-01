# Detector de EPIs em Tempo Real (Visão Computacional)

Este projeto é um sistema de Inteligência Artificial capaz de detetar o uso de Equipamentos de Proteção Individual (EPIs) em tempo real através de uma webcam. 
O modelo identifica múltiplos equipamentos simultaneamente (ex: uma pessoa pode estar a usar capacete e luvas ao mesmo tempo).

## Funcionalidades

* **Deteção em Tempo Real:** Processamento frame a frame via webcam.
* **Classificação Multi-label:** Capaz de identificar várias classes na mesma imagem.
* **Classes Treinadas:**
    * `capacete`
    * `luva`
    * `Oculos`
    * `sem_epi` (nenhum equipamento detetado)
* **Visualização:** Exibe a probabilidade de confiança (%) para cada item detetado no ecrã.

## Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Deep Learning:** TensorFlow / Keras
* **Arquitetura:** MobileNetV2 (Transfer Learning)
* **Visão Computacional:** OpenCV
* **Processamento Numérico:** NumPy

## Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone [https://seu-repositorio-aqui.git](https://seu-repositorio-aqui.git)
cd nome-da-pasta
