import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r*-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ПРЕОБРАЗОВАНИЕ: убедимся, что у нас черные цифры на белом фоне
# В MNIST уже так, но для ясности:
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Изменение формы для CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

# УЛУЧШЕННАЯ АРХИТЕКТУРА CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

cnn_model.summary()

# Компиляция с оптимизатором Adam и снижающимся learning rate
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# АУГМЕНТАЦИЯ ДАННЫХ - КЛЮЧЕВОЕ УЛУЧШЕНИЕ!
datagen = ImageDataGenerator(
    rotation_range=15,           # Повороты ±15 градусов
    width_shift_range=0.15,      # Сдвиг по ширине
    height_shift_range=0.15,     # Сдвиг по высоте
    zoom_range=0.15,             # Масштабирование
    shear_range=0.15,            # Наклон
    fill_mode='nearest'          # Заполнение краев
)

print("Обучение CNN с аугментацией данных...")
start = time.time()

# Обучение с аугментацией
history = cnn_model.fit(
    datagen.flow(x_train_cnn, y_train_ohe, batch_size=128),
    epochs=20,  # Увеличим количество эпох
    validation_data=(x_test_cnn, y_test_ohe),
    verbose=1
)

finish = time.time() - start
print(f"Время обучения: {finish:.2f} секунд")

# Оценка на тестовых данных
test_loss, test_acc = cnn_model.evaluate(x_test_cnn, y_test_ohe, verbose=0)
print(f"\nТочность на тестовых данных: {test_acc*100:.2f}%")

# Сохранение модели
cnn_model.save('CNN.keras')
print("Модель CNN сохранена как 'CNN.keras'")

# Визуализация
plot_history(history)