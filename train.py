import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import salvar_class_indices

def construir_modelo(num_classes, input_shape=(224,224,3), dropout=0.4):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    preds = Dense(num_classes, activation="sigmoid")(x)  # multi-label
    model = Model(inputs=base.input, outputs=preds)

    for layer in base.layers:
        layer.trainable = False
    return model

def main():
    dataset_dir = "dataset"
    output_model = "models/meu_modelo_epi.h5"
    img_size = (224,224)
    batch_size = 16
    epochs = 10

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_gen.class_indices)
    print("Classes:", train_gen.class_indices)

    model = construir_modelo(num_classes, input_shape=(224,224,3))
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(output_model, monitor="val_accuracy", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, reduce_lr])

    salvar_class_indices(train_gen.class_indices, path="class_indices.json")
    print("Modelo salvo em:", output_model)

if __name__ == "__main__":
    main()
