import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder

# Data Pre-Processing
def process_spectral_data(
    root_folder,
    reduce_dimensions=True,
    gaussian_smoothing=True,
    wavelength_range=(0, 1500)
):
    """
    Processes spectral data from the given folder.
    Args:
        root_folder (str): Path to the root folder containing class subdirectories.
        reduce_dimensions (bool): Whether to reduce dimensions by 50% by averaging adjacent points.
        gaussian_smoothing (bool): Whether to apply Gaussian smoothing to intensity data.
        wavelength_range (tuple): Range of wavelengths to consider (start, end).

    Returns:
        np.array(all_data): Matrix of processed data [samples x features].
        np.array(all_labels): Array of integer labels corresponding to each sample.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Find class folders in the root directory
    class_folders = [
        folder for folder in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, folder)) and folder not in ['.', '..']
    ]

    common_wavelengths = np.arange(wavelength_range[0], wavelength_range[1], 1)

    all_data = []
    all_labels = []

    # Iterate over each class folder, find .csv files, apply modifications
    for class_name in class_folders:
        class_path = os.path.join(root_folder, class_name)

        csv_files = [
            file for file in os.listdir(class_path)
            if file.endswith('.csv')
        ]

        if csv_files:
            for file_name in csv_files:
                file_path = os.path.join(class_path, file_name)
                
                # Read file and auto-detect header
                raw_data = pd.read_csv(file_path, header=None)
                first_row = raw_data.iloc[0]

                # Check if the first row is non-numeric
                if not np.issubdtype(first_row.dtypes, np.number):
                    file_data = raw_data.iloc[1:].values  # Skip header
                else:
                    file_data = raw_data.values  # No header, use entire file

                # Ensure data is numeric
                try:
                    file_data = file_data.astype(float)
                except ValueError as e:
                    print(f"Skipping file {file_name}: Unable to convert to numeric. Error: {e}")
                    continue

                wavelengths = file_data[:, 0]
                intensities = file_data[:, 1]

                # Apply Gaussian smoothing if enabled
                if gaussian_smoothing:
                    intensities = savgol_filter(intensities, window_length=11, polyorder=2)

                # Interpolate the intensity data
                interpolated_intensities = np.interp(
                    common_wavelengths, wavelengths, intensities
                )

                # Reduce dimensionality if enabled
                if reduce_dimensions:
                    reduced_intensities = interpolated_intensities.reshape(-1, 2).mean(axis=1)
                else:
                    reduced_intensities = interpolated_intensities

                # Append processed data
                all_data.append(reduced_intensities)
                all_labels.append(class_name)
        else:
            print(f'No .csv files found in folder: {class_name}')

    # Convert labels to integers
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)

    return np.array(all_data), np.array(all_labels)


def prepare_dataset(all_data, all_labels, feature_dim, config):
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_data, all_labels,
        test_size=config["test_ratio"] + config["val_ratio"],
        stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config["test_ratio"] / (config["test_ratio"] + config["val_ratio"]),
        stratify=y_temp
    )

    X_train = X_train.reshape(-1, 1, feature_dim, 1)
    X_val = X_val.reshape(-1, 1, feature_dim, 1)
    X_test = X_test.reshape(-1, 1, feature_dim, 1)

    num_classes = len(np.unique(all_labels))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes