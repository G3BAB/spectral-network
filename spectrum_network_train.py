import tensorflow as tf
import numpy as np
import logging

from utils.config_handler import load_config
from utils.data_processing import process_spectral_data, prepare_dataset
from utils.model_subclasses import SpectralCNN
from utils.training_pipeline import SpectralTrainer, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def main():
    config = load_config()
    set_seed(config["randomizer_seed"])

    log.info("Loading and preprocessing data...")
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

    log.info("Initializing model and trainer...")
    model = SpectralCNN(feature_dim=feature_dim, num_classes=num_classes)
    model.build(input_shape=(None, 1, feature_dim, 1))
    trainer = SpectralTrainer(model, config)
    trainer.compile_model()

    log.info("Starting training...")
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        all_data_shape=all_data.shape,
        X_test_shape=X_test.shape
    )

    log.info("Final evaluation and saving...")
    trainer.finalize(history, X_test, y_test)


if __name__ == "__main__":
    main()
