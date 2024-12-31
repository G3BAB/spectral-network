# Spectral Classification with TensorFlow

This project is designed for the classification of mineral spectra using a Convolutional Neural Network (CNN). It includes tools for processing spectral data, training the CNN model, and evaluating its performance with graphical outputs and a confusion matrix.

## Features

- **Data Processing:** Handles normalization, dimensionality reduction, and noise addition.
- **Customizable Training:** Configuration options for training epochs, learning rates, and dataset ratios via a config file.
- **Evaluation Graphics:** Generates accuracy plots and confusion matrices to assess model performance.

## Acknowledgments

The files included in the default training set are based on the RRUFF database, with modifications such as normalization and noise addition. The author of the repository is not affiliated with RRUFF project.

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
