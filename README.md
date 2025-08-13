# Spectral Classification with TensorFlow

This project is designed for the classification of mineral spectra using a Convolutional Neural Network. It includes tools for processing spectral data, training the CNN model, and evaluating its performance with graphical outputs and a confusion matrix. 

## Usage
- The network accepts raman spectra represented by .csv files (separate file for each sample, with column headers "wavelength" and "intensity").
- In the directory with training data, a folder should be created for each class. The .csv files representing samples from the class should be stored there. The script will take the name of each subdirectory and apply the same name for the corresponding class.
- The config file can be modified to customize the training process. Values present there by default are the ones that gave me the best results for my data.
- The config file will be automatically restored in the event of its deletion, or if it's modified beyond the capabilities of the config handler script.

- **GUI APP** is available in fastapi_test branch - it doesn't function perfectly so far. Collaboration on the app is welcome.

## Future development
As of now, the code trains and evaluates the accuracy of the network. However, a functionality that allows the user to then upload independent data for classification has not yet been implemented. It will be the focus of development in the near future.

Moreover, future work will focus on ensuring more modular and manageable code for better customizability for various data.


## Features

- **Data Preprocessing:** Handles data formatting, dimensionality reduction, and smoothing.
- **Customizable Training:** Convenient configuration options for training epochs, learning rates, dataset ratios and other hyperparameters via a config file.
- **Evaluation mode:** Generates accuracy plots and confusion matrices to assess model performance.
- **Model export:** Model can be exported for later use.

## Acknowledgments

For experimenting and testing the neural networks, RRUFF database of mineral spectra is recommended. The author of the repository is not affiliated with RRUFF project.

More information about RRUFF:

Lafuente B, Downs R T, Yang H, Stone N (2015)  
*The power of databases: the RRUFF project.*  
In: Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter, pp 1-30.  
[Link to the RRUFF project](https://rruff.info/).

## Requirements

The project requires the following dependencies:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

Install all dependencies using:
```bash
pip install -r requirements.txt
