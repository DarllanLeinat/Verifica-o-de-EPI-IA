import json
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def salvar_class_indices(class_indices: dict, path='class_indices.json'):
    with open(path, 'w') as f:
        json.dump(class_indices, f, indent=2, ensure_ascii=False)

def carregar_class_indices(path='class_indices.json'):
    with open(path, 'r') as f:
        return json.load(f)

def carregar_modelo(path='meu_modelo.h5'):
    return load_model(path)

def carregar_e_preprocessar_imagem(path, target_size=(224,224)):

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = preprocess_input(image)  # MobileNetV2 preprocessing
    image = np.expand_dims(image, axis=0)
    return image
