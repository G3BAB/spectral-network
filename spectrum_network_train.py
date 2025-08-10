from utils.config_handler import load_config
from utils.data_processing import process_spectral_data, prepare_dataset
from utils.model_subclasses import SpectralCNN
from utils.training_pipeline import SpectralTrainer, set_seed
from utils.logger import log_training, log_info, log_info_console

# TODO:
# - separate log file for each run, stored in subdirectories with corresponding date
# - save model option in config instead of at runtime

def main():
    config = load_config()
    training_seed = config["randomizer_seed"]
    set_seed(training_seed)

    log_info_console("Loading and preprocessing data...")
    all_data, all_labels = process_spectral_data(
        root_folder=config["root_folder"],
        reduce_dimensions=config["reduce_dimensions"],
        gaussian_smoothing=config["gaussian_smoothing"],
        wavelength_range=config["wavelength_range"]
    )

    feature_dim = all_data.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_dataset(
        all_data, all_labels, feature_dim, config
    )

    log_info_console("Initializing model and trainer...")
    model = SpectralCNN(feature_dim=feature_dim, num_classes=num_classes)
    model.build(input_shape=(None, 1, feature_dim, 1))
    trainer = SpectralTrainer(model, config)
    trainer.compile_model()

    log_training("Starting training...")
    log_training(f"Randomizer seed: {training_seed}")
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        all_data_shape=all_data.shape,
        X_test_shape=X_test.shape
    )

    log_training("Evaluating model...")
    trainer.finalize(history, X_test, y_test)


if __name__ == "__main__":
    main()
