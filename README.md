# 📧 Spam Detection Using LSTM RNN

## 📜 Overview
This project is a **Spam Detection System** using an **LSTM-based Recurrent Neural Network (RNN)**. It classifies messages as either **spam or ham** (not spam) using a dataset of SMS messages.


## 🏗️ Features
- Uses **LSTM (Long Short-Term Memory)** networks for text classification
- Implements **tokenization and padding** for processing text data
- Achieves **98% accuracy** on the test set
- **Smoothed training curves** for better visualization
- Trained on the **SMS Spam Collection Dataset**


## 🗂️ Dataset
The dataset used is the **SMS Spam Collection Dataset**, which consists of labeled messages:
- **ham (0):** Non-spam messages
- **spam (1):** Unwanted/spam messages


## 🛠️ Installation & Setup
To run this project, install the required dependencies:

```bash
pip install numpy pandas tensorflow matplotlib scikit-learn
Run the Training Script:

bash
Run
Copy code
python train.py
This will train the model and save the weights as spam_classifier_weights.weights.h5.

Run the Testing Script:

bash
Run
Copy code
python test.py
This will load the trained model and test it on sample messages.

🧑‍💻 Model Architecture
```

## 🛠 Running the code.
To run this project, install the required dependencies:





