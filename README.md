Face Detection DB-VAE Demo

This repository contains a Debiasing Variational Autoencoder (DB-VAE) for facial detection, along with a Streamlit web application for demo purposes. The model aims to detect faces in images while mitigating potential biases in skin tone and gender.

📌 Project Overview

Modern facial detection models often show bias due to imbalanced training datasets. The DB-VAE approach leverages a variational autoencoder to learn a latent representation of facial features, which allows adaptive resampling of underrepresented features (e.g., dark-skinned faces, faces with accessories).

Key features:

Debiased facial detection using latent space representation.

Uses a combination of CelebA (positive face images) and ImageNet (negative examples) datasets.

Streamlit demo for uploading images and visualizing face detection probability.

🛠 Technologies & Libraries

Python 3.10+

PyTorch

torchvision

NumPy

Matplotlib

MIT Deep Learning library

Streamlit

📝 Repository Structure
├── app.py                         # Streamlit app
├── Debiasing_Facial_Detection_Systems.py   # DB-VAE model definitions
├── dbvae_weights.pth               # Trained model weights
├── requirements.txt                # Python dependencies
└── README.md

How to Run this :
https://face-detection-dbvae-gdh9nuy7detkrbp4hz72ob.streamlit.app/

Upload an image in the web interface to see the probability of a face.

📈 Using the DB-VAE Model

The DB-VAE model (DB_VAE) contains:

Encoder: CNN that outputs classification logits + latent mean & log-variance.

Reparameterization trick: Samples latent vector from learned distribution.

Decoder: Reconstructs images from latent vector.

During training, adaptive resampling increases the probability of rare latent features to reduce bias.

⚙️ Deployment

The app is ready for deployment on Streamlit Cloud
.

Make sure to include all files (app.py, Debiasing_Facial_Detection_Systems.py, dbvae_weights.pth) in your GitHub repo.

Ensure requirements.txt lists all necessary dependencies.

👤 Author

Ahmed Khaled
Email: zakreahmed666@example.com

CelebA Dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

MIT Deep Learning Library: https://github.com/mitdeeplearning/mitdeeplearning
