# spectrum_network_train.py

from utils.config_handler import Config
from utils.data_processing import process_spectral_data, prepare_dataset
from utils.model_subclasses import SpectralCNN
from utils.training_pipeline import SpectralTrainer, set_seed
from utils.logger import log_training, log_info_console

# SHARED STATUS (READ BY /status AND /results)
training_status = {
    "running": False,
    "epoch": 0,
    "loss": None,
    "val_loss": None,
    "total_epochs": 0,
    "final_accuracy": None,
    "confusion_matrix": None
}

def run_training(config: Config):
    training_status["running"] = True
    training_status["epoch"] = 0
    training_status["loss"] = None
    training_status["val_loss"] = None
    training_status["total_epochs"] = config.epochs  # <-- Set total epochs here
    training_status["final_accuracy"] = None
    training_status["confusion_matrix"] = None

    set_seed(config.randomizer_seed)

    log_info_console("Loading and preprocessing data...")
    all_data, all_labels = process_spectral_data(
        root_folder=config.root_folder,
        reduce_dimensions=config.reduce_dimensions,
        gaussian_smoothing=config.gaussian_smoothing,
        wavelength_range=config.wavelength_range
    )

    feature_dim = all_data.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_dataset(
        all_data, all_labels, feature_dim, config
    )

    log_info_console("Initializing model and trainer...")
    model = SpectralCNN(feature_dim=feature_dim, num_classes=num_classes)
    model.build(input_shape=(None, 1, feature_dim, 1))
    trainer = SpectralTrainer(model, config)

    # PROGRESS CALLBACK FOR EACH EPOCH (CALL THIS INSIDE trainer.train)
    def update_status(epoch, loss, val_loss):
        training_status["epoch"] = epoch + 1  # Keras uses 0-based epoch
        training_status["loss"] = loss
        training_status["val_loss"] = val_loss

    trainer.compile_model()

    log_training("Starting training...")
    log_training(f"Randomizer seed: {config.randomizer_seed}")

    # Ensure your trainer.train calls status_callback at end of each epoch
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        all_data_shape=all_data.shape,
        X_test_shape=X_test.shape,
        status_callback=update_status  # <-- add this param in your trainer signature
    )

    log_training("Evaluating model...")
    # CHANGE: finalize returns (test_accuracy, confusion_matrix)
    test_acc, conf_mat = trainer.finalize(history, X_test, y_test)

    # Normalize conf_mat to a plain list for JSON
    try:
        conf_mat = conf_mat.tolist()
    except AttributeError:
        pass

    training_status["final_accuracy"] = float(test_acc) if test_acc is not None else None
    training_status["confusion_matrix"] = conf_mat
    training_status["running"] = False


if __name__ == "__main__":
    cfg = Config.from_file()
    run_training(cfg)
