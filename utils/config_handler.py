import os
import sys
import logging

CONFIG_FILE = "config.txt"
DEFAULT_CONFIG = {
    "epochs": 100,
    "initial_learning_rate": 0.0115,
    "learning_rate_scaling": 0.975,
    "scaling_frequency": 10,
    "validation_patience": 20,
    "batch_size": 64,
    "test_ratio": 0.22,
    "val_ratio": 0.15,
    "root_folder": "./data/TRAIN",
    "model_save_path": "model/model.keras",
    "reduce_dimensions": True,
    "gaussian_smoothing": True,
    "wavelength_range": (0, 1500),
    "evaluation_graphics": True,
    "randomizer_seed": 17
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def load_config(config_file=CONFIG_FILE):
    """Loads configuration parameters from a file, with validation and user intervention for missing values."""

    if not os.path.exists(config_file):
        logging.warning(f"{config_file} not found. Regenerating with default values.")
        regenerate_config(config_file, DEFAULT_CONFIG)

    config = {}
    missing_keys = []

    with open(config_file, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            try:
                if value.lower() in ["true", "false"]:
                    config[key] = value.lower() == "true"
                elif "," in value and key == "wavelength_range":
                    config[key] = tuple(map(int, value.strip("()").split(",")))
                else:
                    config[key] = float(value) if "." in value else int(value)
            except ValueError:
                config[key] = value

    # Check for missing keys
    for key in DEFAULT_CONFIG.keys():
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        logging.warning(f"\nMissing configuration keys: {', '.join(missing_keys)}")
        response = input("\nChoose an action:\n[1] Terminate script [2] Use default values and proceed: ")
        if response == "1":
            sys.exit("Terminating due to missing config values. Modify the config file and try again.")
        elif response == "2":
            for key in missing_keys:
                config[key] = DEFAULT_CONFIG[key]

            save_config(config, config_file)
            logging.info("Loaded default values")

        else:
            sys.exit("Invalid input. Terminating script.")

    check_ratios(config)
    return config


def regenerate_config(config_file=CONFIG_FILE, defaults=DEFAULT_CONFIG):
    """Restores the config file with default values determined at the start of this module."""

    with open(config_file, "w") as file:
        for key, value in defaults.items():
            file.write(f"{key}={value}\n")
    logging.info(f"{config_file} has been reset to default values.")


def save_config(config, config_file=CONFIG_FILE):
    """Saves the configuration to a file."""
    with open(config_file, "w") as file:
        for key, value in config.items():
            file.write(f"{key}={value}\n")
    logging.info(f"Configuration saved to {config_file}.")


def check_ratios(config):
    """Validates that dataset split ratios make sense."""

    train_ratio = 1 - (config["test_ratio"] + config["val_ratio"])
    if config["test_ratio"] < 0 or config["val_ratio"] < 0:
        sys.exit("Error: Test and validation ratios cannot be negative.")
    if train_ratio <= 0:
        sys.exit("Error: Invalid configuration. Train ratio is non-positive.")
    if train_ratio < 0.5:
        logging.warning(f"Train ratio is below 0.5!!! (currently {train_ratio:.2f}).")
