import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from CNN import plot_history  # Импорт функции визуализации

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

# УЛУЧШЕННАЯ АРХИТЕКТУРА MLP
mlp_model = Sequential([
    Flatten(input_shape=(28, 28)),
    
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(10, activation='softmax')
])

mlp_model.summary()

# Компиляция
mlp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks для улучшения обучения
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("Обучение MLP...")
start = time.time()

# Обучение
history = mlp_model.fit(
    x_train, y_train_ohe,
    epochs=50,  # Больше эпох, но EarlyStopping остановит раньше если нужно
    batch_size=128,
    validation_split=0.1,  # Используем 10% тренировочных данных для валидации
    callbacks=callbacks,
    verbose=1
)

finish = time.time() - start
print(f"Время обучения: {finish:.2f} секунд")

# Окончательная оценка на тестовых данных
test_loss, test_acc = mlp_model.evaluate(x_test, y_test_ohe, verbose=0)
print(f"\nТочность на тестовых данных: {test_acc*100:.2f}%")

# Сохранение модели
mlp_model.save('MLP_optimized.keras')
print("Модель MLP сохранена как 'MLP_optimized.keras'")

# Визуализация
plot_history(history)