# 🛡️ Detetor de EPIs em Tempo Real

Este projeto é um sistema de Inteligência Artificial capaz de detetar o uso de Equipamentos de Proteção Individual (EPIs) em tempo real através de uma webcam. 

O modelo identifica múltiplos equipamentos simultaneamente (ex.: uma pessoa pode estar a usar capacete e luvas ao mesmo tempo).

## Funcionalidades

- **Deteção em Tempo Real:** Processamento fotograma a fotograma via webcam
- **Classificação Multi-label:** Capaz de identificar várias classes na mesma imagem
- **Classes Treinadas:**
  - `capacete`
  - `luva`
  - `óculos`
  - `sem_epi` (nenhum equipamento detetado)
- **Visualização:** Exibe a probabilidade de confiança (%) para cada item detetado no ecrã

## Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **Arquitetura:** MobileNetV2 (Transfer Learning)
- **Visão Computacional:** OpenCV
- **Processamento Numérico:** NumPy

## Instalação e Configuração

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Configurar o Ambiente

Recomenda-se a criação de um ambiente virtual para isolar as dependências:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

Com o ambiente ativo, instale todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Como Usar

### Opção A: Executar a Deteção (Webcam)

Para iniciar a deteção em tempo real usando o modelo já treinado:

```bash
python camera.py
```

- A webcam será aberta
- O sistema desenhará caixas de texto com a percentagem de certeza
- Pressione a tecla **'q'** para encerrar o programa

### Opção B: Treinar um Novo Modelo

Caso queira treinar o modelo do zero com as suas próprias imagens:

**1. Organize o Dataset:** 

Crie uma pasta chamada `dataset` na raiz do projeto e organize as imagens em subpastas:

```
dataset/
├── capacete/
├── luva/
├── óculos/
└── sem_epi/
```

**2. Inicie o Treino:**

```bash
python train.py
```

- O script fará o *Fine-Tuning* na MobileNetV2
- O novo modelo será guardado como `models/meu_modelo_epi.h5`

## Estrutura do Projeto

```
.
├── camera.py                    # Script principal de inferência (webcam)
├── train.py                     # Script de treino da rede neuronal
├── utils.py                     # Funções auxiliares
├── requirements.txt             # Dependências do projeto
├── class_indices.json           # Mapeamento de classes (Label → ID)
├── models/
│   └── meu_modelo_epi.h5       # Modelo treinado
└── dataset/                     # Pasta com imagens de treino
    ├── capacete/
    ├── luva/
    ├── óculos/
    └── sem_epi/
```

## Detalhes da Arquitetura

O projeto aplica **Transfer Learning** na rede **MobileNetV2**:

1. **Input:** Imagens redimensionadas para 224×224 píxeis
2. **Feature Extraction:** Usa os pesos pré-treinados da ImageNet (camadas congeladas)
3. **Classifier Head:**
   - `GlobalAveragePooling2D`
   - `Dropout (0.4)` (Regularização)
   - `Dense (Sigmoid)` (Saída)
4. **Loss Function:** `Binary Crossentropy` (ideal para classificação multi-label)

## Requisitos do Sistema

- Python 3.7+
- Webcam funcional
- 4GB RAM (mínimo recomendado)
- GPU (opcional, mas recomendado para treino)

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abrir um Pull Request

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!
