import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.logger import log_training, log_info, log_info_console, log_warning, log_error


class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, dataset_shapes):
        super().__init__()
        self.total_epochs = total_epochs
        self.dataset_shapes = dataset_shapes

    def on_train_begin(self, logs=None):
        train_size, val_size, test_size, all_size = self.dataset_shapes
        log_info(f"Total samples: {all_size}")
        log_info(f"Training samples: {train_size}")
        log_info(f"Test samples: {test_size}")
        log_info(f"Validation samples: {val_size}")
        log_info("Starting training...\n")

    def on_epoch_end(self, epoch, logs=None):
        metrics = [f"{key}: {value:.4f}" for key, value in logs.items()]
        metrics_str = f"Epoch {epoch + 1}/{self.total_epochs} - " + " - ".join(metrics)
        log_info(metrics_str)

    def on_train_end(self, logs=None):
        log_info_console("Training finished.")


class SpectralTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.initial_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, all_data_shape, X_test_shape):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.validation_patience,
            restore_best_weights=True
        )

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: self.config.initial_learning_rate * (
                self.config.learning_rate_scaling ** (epoch // self.config.scaling_frequency)
            )
        )

        print_logs = PrintLogs(
            total_epochs=self.config.epochs,
            dataset_shapes=(
                X_train.shape[0],
                X_val.shape[0],
                X_test_shape[0],
                all_data_shape[0]
            )
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping, lr_scheduler, print_logs],
            verbose=0
        )
        return history

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        log_info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_accuracy

    def plot_results(self, history, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
        log_info_console("Confusion Matrix:\n%s", confusion_mtx)

        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def maybe_save(self):
        save_model = input("\nWould you like to save the trained model? (y/n): ").strip().lower()
        if save_model in ["yes", "y"]:
            save_path = self.config.model_save_path
            self.model.save(save_path)
            log_info(f"Model saved to: {save_path}")
        else:
            log_info("Model was not saved.")

    def finalize(self, history, X_test, y_test):
        self.evaluate(X_test, y_test)
        print(self.config.evaluation_graphics)
        if self.config.evaluation_graphics==True:
            self.plot_results(history, X_test, y_test)
        self.maybe_save()


def set_seed(seed=1):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
