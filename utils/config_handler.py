import os
import sys
from utils.logger import log_info_console, log_warning


CONFIG_FILE = "config.txt"
DEFAULT_CONFIG = {
    "epochs": 75,
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

class Config:
    """
    Represents the application configuration.

    Loads from a file, stores config values as attributes, validates them,
    and can save back to disk. Missing keys are filled with DEFAULT_CONFIG.
    """

    def __init__(self, **kwargs):
        # Populate attributes from kwargs or fall back to defaults
        for key, default in DEFAULT_CONFIG.items():
            setattr(self, key, kwargs.get(key, default))

        self._validate()

    @classmethod
    def from_file(cls, path=CONFIG_FILE):
        """
        Create a Config instance from a config file.
        If file is missing, regenerate with defaults.
        """
        if not os.path.exists(path):
            log_warning(f"{path} not found. Regenerating with default values.")
            cls._regenerate_file(path)
            return cls(**DEFAULT_CONFIG)

        loaded = {}
        
        # Read and parse each line of the config file
        with open(path, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                loaded[key] = cls._parse_value(key, value)

        # Fill in any missing keys with defaults
        for key, default in DEFAULT_CONFIG.items():
            if key not in loaded:
                log_warning(f"Missing '{key}' in config, using default value.")
                loaded[key] = default

        config = cls(**loaded)
        config.save(path)
        return config

    @staticmethod
    def _parse_value(key, value):
        """
        Convert string values from the config file into correct Python types.
        Handles bools, tuples, floats, and ints.
        """

        # Boolean parsing
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        
        # Special case for wavelength_range tuple
        if key == "wavelength_range" and "," in value:
            return tuple(map(int, value.strip("()").split(",")))
        
        if "." in value:
            try:
                return float(value)
            except ValueError:
                pass
        
        try:
            return int(value)
        except ValueError:
            return value

    def save(self, path=CONFIG_FILE):
        """
        Write the current config values back to a file.
        """
        with open(path, "w") as f:
            for key in DEFAULT_CONFIG:
                f.write(f"{key}={getattr(self, key)}\n")
        log_info_console(f"Configuration saved to {path}.")

    @staticmethod
    def _regenerate_file(path):
        """
        Write a fresh config file containing DEFAULT_CONFIG values.
        """
        with open(path, "w") as f:
            for key, val in DEFAULT_CONFIG.items():
                f.write(f"{key}={val}\n")
        log_info_console(f"{path} regenerated with default values.")

    def _validate(self):
        """
        Run sanity checks on configuration values.
        Exits the program if critical values are invalid.
        """
        train_ratio = 1 - (self.test_ratio + self.val_ratio)

        if self.test_ratio < 0 or self.val_ratio < 0:
            sys.exit("Error: Test and validation ratios cannot be negative.")
        if train_ratio <= 0:
            sys.exit("Error: Invalid configuration. Train ratio is non-positive.")
        if train_ratio < 0.5:
            log_warning(f"Train ratio is below 0.5!!! (currently {train_ratio:.2f}).")
