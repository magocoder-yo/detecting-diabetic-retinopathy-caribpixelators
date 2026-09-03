# Detecting Diabetic Retinopathy with Machine Learning
#### By CaribPixelators (magocoder-yo, codererufu, dandabot2685, zxch246)

This repository contains various files related to the project. *Workshop_Project.ipynb* is the primary Jupyter notebook that we used during development to create the model. *website.py* contains the code for running the Streamlit website (a live demo of the application viewable here: https://detecting-diabetic-retinopathy-caribpixelators.streamlit.app/). *retinamnist_64.npz* is the data file we used from the RetinaMNIST dataset, *CNN_Model.keras* stores the weights and biases of our trained model that can be used to make predictions, *requirements.txt* contains project dependency information, and *.streamlit/config.toml* contains configuration information for the Streamlit web app.

The goal of the project is to detect the various stages of diabetic retinopathy and build a healthier Caribbean society driven by A.I-powered image processing systems. It uses a convolutional neural network model in TensorFlow (the code is in *Workshop_Project.ipynb*), accessible via the Streamlit web app.

The data used to train the model (*retinamnist_64.npz*) belongs to the MedMNIST dataset (see https://medmnist.com/), obtained from their Zenodo page (see https://zenodo.org/records/10519652/files/retinamnist_64.npz)

The target audience of the project is medical institutions and universities (research & development).

The project is currently a work in progress, it is under active development.