# spectrum_network_train.py

from utils.config_handler import Config
from utils.data_processing import process_spectral_data, prepare_dataset
from utils.model_subclasses import SpectralCNN
from utils.training_pipeline import SpectralTrainer, set_seed
from utils.logger import log_training, log_info, log_info_console

# NEW: GLOBAL TRAINING STATUS FOR PROGRESS REPORTING (USED BY FASTAPI ENDPOINTS)
training_status = {
    "running": False,
    "epoch": 0,
    "loss": None,
    "val_loss": None
}

# CHANGED: main() → run_training(config) SO IT CAN BE CALLED FROM FASTAPI
def run_training(config: Config):
    # UPDATE STATUS AT START
    training_status["running"] = True
    training_status["epoch"] = 0
    training_status["loss"] = None
    training_status["val_loss"] = None

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

    # NEW: PASS CALLBACK OR STATUS UPDATER INTO TRAINER FOR LIVE STATUS
    def update_status(epoch, loss, val_loss):
        training_status["epoch"] = epoch
        training_status["loss"] = float(loss)
        training_status["val_loss"] = float(val_loss)

    trainer.compile_model()

    log_training("Starting training...")
    log_training(f"Randomizer seed: {config.randomizer_seed}")

    # CHANGED: TRAINING FUNCTION RECEIVES UPDATE_STATUS SO PROGRESS IS TRACKED
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        all_data_shape=all_data.shape,
        X_test_shape=X_test.shape,
        status_callback=update_status  # YOU’LL ADD THIS ARG TO trainer.train()
    )

    log_training("Evaluating model...")
    trainer.finalize(history, X_test, y_test)

    # UPDATE STATUS AT END
    training_status["running"] = False


# CHANGED: RETAIN OLD SCRIPT ENTRY FOR MANUAL EXECUTION
if __name__ == "__main__":
    cfg = Config.from_file()
    run_training(cfg)
