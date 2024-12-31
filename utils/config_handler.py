import os
import sys

CONFIG_FILE = "config.txt"
DEFAULT_CONFIG = {
    "epochs": 600,
    "initial_learning_rate": 0.0115,
    "learning_rate_scaling": 0.975,
    "scaling_frequency": 10,
    "validation_patience": 200,
    "batch_size": 64,
    "test_ratio": 0.22,
    "val_ratio": 0.15,
    "root_folder": "./data/TRAIN",
    "model_save_path":"model/model.keras",
    "reduce_dimensionality": True,
    "gaussian_smoothing": True,
    "wavelength_range": (0, 1500),
    "evaluation_graphics": True
}

def load_config(config_file=CONFIG_FILE):
    """Loads configuration parameters from a file, with validation and user intervention for missing values."""
    if not os.path.exists(config_file):
        print(f"\n{config_file} not found. Generating a default configuration file.")
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

    # Validate configuration
    for key in DEFAULT_CONFIG.keys():
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        print(f"\nMissing configuration  in config file: {', '.join(missing_keys)}")
        response = input("Choose an action: [1] Terminate script [2] Use default values and proceed: ")
        if response == "1":
            sys.exit("Terminating due to missing configuration values. Modify the configuration file and try again.")
        elif response == "2":
            for key in missing_keys:
                config[key] = DEFAULT_CONFIG[key]
            save_config(config, config_file)
            print("Missing values replaced with defaults.")
        else:
            sys.exit("Invalid input. Terminating script. Modify the configuration file or rerun the script to restore defaults.")

    check_ratios(config)
    print("\n")
    return config


def regenerate_config(config_file=CONFIG_FILE, defaults=DEFAULT_CONFIG):
    """Restores the config file in case of any mismatch"""
    with open(config_file, "w") as file:
        for key, value in defaults.items():
            if isinstance(value, tuple):
                file.write(f"{key}={value}\n")
            else:
                file.write(f"{key}={value}\n")
    print(f"{config_file} has been reset to default values.")


def save_config(config, config_file=CONFIG_FILE):
    with open(config_file, "w") as file:
        for key, value in config.items():
            if isinstance(value, tuple):
                file.write(f"{key}={value}\n")
            else:
                file.write(f"{key}={value}\n")
    print(f"Configuration saved to {config_file}.")


def check_ratios(config):
    train_ratio = 1 - (config["test_ratio"] + config["val_ratio"])
    if config["test_ratio"] < 0 or config["val_ratio"] < 0:
        sys.exit("Error: Test and validation ratios cannot be negative.")
    if train_ratio < 0.5:
        print(f"Warning: Train ratio is below 0.5 (currently {train_ratio:.2f}).")
    if train_ratio <= 0:
        sys.exit("Error: Invalid configuration. Train ratio is non-positive.")
