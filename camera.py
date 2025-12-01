import cv2
import numpy as np
from utils import carregar_modelo, carregar_class_indices
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def main():
    model = carregar_modelo("models/meu_modelo_epi.h5")
    class_indices = carregar_class_indices("class_indices.json")
    inv_map = {v: k for k, v in class_indices.items()}

    # abrir câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # pré-processar frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # predição
        preds = model.predict(img, verbose=0)[0]

        # mostrar todas as classes com porcentagem
        y_offset = 40
        for idx, conf in enumerate(preds):
            nome = inv_map[idx]
            porcentagem = conf * 100
            texto = f"{nome}: {porcentagem:.1f}%"

            cor = (0, 255, 0) if conf >= 0.5 else (0, 0, 255)
            cv2.putText(frame, texto, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
            y_offset += 40

        # exibir vídeo
        cv2.imshow("Detecção de EPIs (Capacete, Luva, Óculos) - Pressione Q para sair", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
