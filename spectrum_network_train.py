import tensorflow as tf
tf.autograph.set_verbosity(3)
from sklearn.model_selection import train_test_split
import numpy as np
from utils.process_spectral_data import process_spectral_data
from utils.config_handler import load_config
import sys

# Configuration values are stored in config.txt file
config = load_config()

# Custom callbacks replacing default Keras ones
class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        print(f"\nTotal number of samples: {all_data.shape[0]}")
        print(f"Number of training samples: {X_train.shape[0]}")
        print(f"Number of test samples: {X_test.shape[0]}")
        print(f"Number of validation samples: {X_val.shape[0]}")
        print("\nStarting training...\n")

    def on_epoch_end(self, epoch, logs=None):
        metrics = [f"{key}: {value:.4f}" for key, value in logs.items()]
        metrics_str = " - ".join(metrics)
        progress = f"Epoch {epoch + 1}/{self.total_epochs} - {metrics_str}\r"
        sys.stdout.write(progress)
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()


def prepare_dataset(all_data, all_labels, feature_dim, config):
    """Divides dataset into train, test and validation sets in accordance to 
    config.txt file. Reshapes the data and uses one-hot encoding to ensure
    compliance with the CNN architecture."""

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_data, all_labels, test_size=config["test_ratio"] + config["val_ratio"], stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config["test_ratio"] / (config["test_ratio"] + config["val_ratio"]), stratify=y_temp
    )

    X_train = X_train.reshape(-1, 1, feature_dim, 1)
    X_val = X_val.reshape(-1, 1, feature_dim, 1)
    X_test = X_test.reshape(-1, 1, feature_dim, 1)

    # One-hot encoding
    num_classes = len(np.unique(all_labels))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


# --------------------------------------------------------------
# Data Pre-Processing and Preparation
# --------------------------------------------------------------
all_data, all_labels = process_spectral_data(
    root_folder=config["root_folder"], 
    reduce_dimensionality=config["reduce_dimensionality"], 
    gaussian_smoothing=config["gaussian_smoothing"], 
    wavelength_range=config["wavelength_range"]
)

feature_dim = all_data.shape[1]

X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_dataset(
    all_data, all_labels, feature_dim, config
)

# --------------------------------------------------------------
# CNN Definition
# --------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1, feature_dim, 1)),
    tf.keras.layers.Conv2D(64, (1, 8), padding='same', activation='gelu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),

    tf.keras.layers.Conv2D(64, (1, 8), padding='same', activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["initial_learning_rate"]),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------------------------------------
# Training
# --------------------------------------------------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=config["validation_patience"], restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: config["initial_learning_rate"] * (config["learning_rate_scaling"] ** (epoch // config["scaling_frequency"]))
)

print_logs = PrintLogs(total_epochs=config["epochs"])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[early_stopping, lr_scheduler, print_logs],
    verbose=0
)

# --------------------------------------------------------------
# Testing and Evaluation
# --------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy (Test Set): {test_accuracy * 100:.2f}%\n")

if config.get("evaluation_graphics", False):
    # Confusion Matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
    print("Confusion Matrix:")
    print(confusion_mtx[:])

    # Accuracy over epochs
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Save Model
# --------------------------------------------------------------
save_model = input("\nWould you like to save the trained model? (yes/no): ").strip().lower()
if save_model in ["yes", "y"]:
    save_path = config.get("model_save_path", "saved_model")
    model.save(save_path)
    print(f"Model saved to: {save_path}")
else:
    print("Model was not saved.")
