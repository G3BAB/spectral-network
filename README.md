# Spectral Classification with TensorFlow

This project is designed for the classification of mineral spectra using a Convolutional Neural Network. It includes tools for processing spectral data, training the CNN model, and evaluating its performance with graphical outputs and a confusion matrix. Example data for training and testing is bundled with the repository, but can be replaced with other data.

## Future development
As of now, the code trains and evaluates the accuracy of the network. However, a functionality that allows the user to then upload independent data for classification has not yet been implemented. It will be the focus of development in the near future.

## Features

- **Data Processing:** Handles data formatting, dimensionality reduction, and smoothing.
- **Customizable Training:** Convenient configuration options for training epochs, learning rates, dataset ratios and other hyperparameters via a config file.
- **Evaluation mode:** Generates accuracy plots and confusion matrices to assess model performance.
- **Model export:** Model can be exported for later use.

## Acknowledgments

The files included in the default training set are based on the RRUFF database, with modifications such as normalization and noise addition. The author of the repository is not affiliated with RRUFF project.

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
