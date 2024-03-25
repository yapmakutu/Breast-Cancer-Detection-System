# Introduction

This project is a graduation endeavor aimed at developing a computer-aided diagnostic (CAD) system for the detection of breast cancer from ultrasound images. Leveraging advanced deep learning models, this system seeks to provide an efficient and accurate tool for medical professionals.

## Model Download and Configuration Setup

This CAD system depends on specific deep learning models. To make the process of downloading models easier, we have automated the download. A script called 'Models.py' was created for this purpose. It automatically downloads the required models from Google Drive and updates the config.json file with the appropriate file paths. Before starting the main application, make sure to run 'Models.py' to ensure all models are installed correctly.

### Prerequisites

- Python 3.8 or higher is required to run this project.

## Installation

To prepare the project environment, follow these steps:

1. **Install Required Libraries**: Ensure all necessary Python libraries are installed by running the following command in your terminal or command prompt: pip install -r requirements.txt

This command reads the `requirements.txt` file and installs all listed libraries.

2. **Download Models**: Run the `Models.py` script to download the required models and to automatically update the `config.json` with their file paths: python Models.py

This step is crucial for the proper execution of the project as it ensures all models are correctly downloaded and configured.

## Running the Application

After completing the installation and model setup, you can start the project by running the `MainApplication` file. This initiates the CAD system, ready to process ultrasound images for breast cancer detection.






