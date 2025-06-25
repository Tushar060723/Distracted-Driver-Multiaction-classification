# Distracted-Driver-Multiaction-classification

## Table of Contents
* [General Information](#general-information)
  * [Overview](#overview)
  * [Dataset](#dataset)
  * [Objective](#objective)
  * [Methodology](#methodology)
* [Technologies Used](#technologies-used)
  * [Key Libraries and Their Roles](#key-libraries-and-their-roles)
    * [Natural Language Processing (NLP) Libraries](#natural-language-processing-nlp-libraries)
    * [Machine Learning and Deep Learning Libraries](#machine-learning-and-deep-learning-libraries)
    * [Data Visualization Libraries](#data-visualization-libraries)
* [Conclusions](#conclusions)
* [Project Files](#project-files)

## General Information

### Overview
This project aims to classify images into 10 categories representing different driver behaviors, such as texting, talking on the phone, and safe driving, among others.

### Dataset
The dataset can be downloaded [here](https://www.dropbox.com/s/0vyzjcqsdl6cqi2/state-farm-distracted-driver-detection.zip?dl=0).

### Objective
To develop a model capable of accurately classifying driver behaviors from images, with classes including safe driving, texting, talking on the phone, and more.

### Methodology
1. Data was organized into training and test directories, with images categorized into subdirectories based on their respective classes.
3. Preprocessing involved creating a validation directory by transferring 20% of images from each class to corresponding subdirectories under the validation directory, ensuring balanced representation for model training and validation. This step was executed in a different notebook `preprocessing.ipynb`, which is also a part of current repo.
4. Exploratory Data Analysis (EDA) was conducted to understand the distribution of images across categories and subjects.
5. A Convolutional Neural Network (CNN) model was developed using transfer learning from MobileNet.
6. The model architecture included a GlobalAveragePooling2D layer followed by dense layers with ReLU activation and dropout regularization.
7. Model performance was evaluated on the validation set, achieving an accuracy of 86.16%.
8. Predictions were made on the test set, and a submission.csv file was generated listing the predicted probabilities for each class associated with each test image.

## Technologies Used

This project leverages a variety of technologies, libraries, and frameworks:

- **Conda**: 23.5.2
- **Python**: 3.8.18
- **NumPy**: 1.22.3
- **Pandas**: 2.0.3
- **Matplotlib**:
  - Core: 3.7.2
  - Inline: 0.1.6
- **Seaborn**: 0.12.2
- **Scikit-learn**: 1.3.0
- **TensorFlow and Related Packages**:
  - TensorFlow Dependencies: 2.9.0
  - TensorFlow Estimator: 2.13.0 (via PyPI)
  - TensorFlow for macOS: 2.13.0 (via PyPI)
  - TensorFlow Metal: 1.0.1 (for GPU acceleration on macOS, via PyPI)
  - keras - 2.13.1 

### Key Libraries and Their Roles:

#### Machine Learning and Deep Learning Libraries:
- **Scikit-learn**: Provides tools for data preprocessing, model selection, and evaluation metrics, supporting a wide range of machine learning tasks.
- **TensorFlow (including Keras)**: The backbone for building and training advanced neural network models, including GRU and Bidirectional GRU architectures, to handle complex text classification challenges.

#### Data Visualization Libraries:
- **Matplotlib** and **Seaborn**: Integral for creating a wide array of data visualizations, from simple plots to complex heatmaps, to analyze model performance and explore data characteristics.


## Conclusions
- Our CNN model demonstrated promising results in classifying driver behaviors from images, achieving an accuracy of 86.16% on the validation set.
- Despite the solid performance, there remains ample opportunity for model enhancement and refinement to further improve accuracy and robustness.

## Project Files
- `preprocessing.ipynb`: Jupyter notebook containing data preprocessing steps, including the creation of the validation directory.
- `model_training.ipynb`: Jupyter notebook containing CNN model development, training, and evaluation.
