
# Cucumber Leaf Disease and Nutrient Deficiency Detection

This repository contains the deep learning model training pipeline and saved weights for diagnosing diseases and nutrient deficiencies in cucumber plants. The model is designed to be lightweight and efficient, serving as the backend intelligence for a cross-platform mobile application (React Native / Expo).

## Project Overview

Early and accurate detection of plant diseases and nutrient deficiencies is crucial for crop yield and quality. This project utilizes a **MobileNetV4** architecture, implemented in PyTorch, to classify images of cucumber leaves into various health categories. MobileNetV4 was specifically chosen for its optimal balance between accuracy and computational efficiency, making it ideal for on-device or edge-based mobile inference.

### Key Features
 **State-of-the-Art Architecture:** Leverages MobileNetV4 via the `timm` library for efficient feature extraction.
 **Advanced Optimizer:** Utilizes the `lion-pytorch` optimizer for potentially faster convergence and better generalization compared to standard AdamW.
 **Robust Augmentation:** Employs `albumentations` for extensive data augmentation to improve model robustness under varying real-world lighting and camera conditions.
 **Mobile-Ready:** The final `.pt` model is optimized for integration with a React Native mobile application frontend.

## Repository Structure


├── CUCUMBER_TRAINING.IPYNB       # Jupyter Notebook containing the full training pipeline
├── mobilenetv4_cucumber_best.pt  # PyTorch model weights of the best performing epoch
├── requirements.txt              # Project dependencies
├── .gitignore                    # Specifies intentionally untracked files to ignore
└── README.md                     # Project documentation


## Installation & Usage

To run the training notebook locally or in a cloud environment (like Google Colab), follow these steps:

**1. Clone the repository:**


**2. Set up your environment:**
Open `CUCUMBER_TRAINING.IPYNB` in your preferred environment, such as Visual Studio Code, Jupyter Notebook, or Google Colab.

**3. Install Dependencies:**
Ensure you have PyTorch installed for your specific hardware configuration. You will also need the additional libraries used in the notebook. You can install them in your terminal via:

pip install torch torchvision timm lion-pytorch albumentations kagglehub tqdm

*(Alternatively, you can install these via a `requirements.txt` file if you prefer: `pip install -r requirements.txt`)*

**4. Run the Notebook:**
Change the filepaths to your corresponding/required filepahts. Execute the cells in the notebook sequentially. The code will automatically:
* Download the dataset via Kaggle.
* Apply data augmentations.
* Initialize and train the MobileNetV4 model.
* Save the resulting weights locally.
