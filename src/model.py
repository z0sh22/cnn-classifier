from tensorflow import keras
from tensorflow.keras import layers

def build_simple_cnn(input_shape, num_classes=10):
    model = keras.Sequential([
        layers.Input(shape = input_shape),

        # Первый слой (простые паттерны)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), 

        # Второй слой (текстуры формы)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), 

        # Финал классификация
        layers.Flatten(), 
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),

    ], name="simple_cnn")
    return model

def build_medium_cnn(input_shape, num_classes=10):

    model = keras.Sequential([
        
        layers.Input(shape=input_shape),
        
        # Первый слой
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        
        # Второй слой
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Третий слой
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="medium_cnn")

    return model 

def build_deep_cnn(input_shape, num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Первый
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        
        # Второй 
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Третий 
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        
        # Четвертый 
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Пятый
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),

    ], name="deep_cnn")
    return model

ARCHITECTURES = {
    "simple": build_simple_cnn,
    "medium": build_medium_cnn,
    "deep":   build_deep_cnn,
}