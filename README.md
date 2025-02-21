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
```
#### Run the Training Script:
```bash
Run👇
python train.py
```
#### This will train the model and save the weights as spam_classifier_weights.weights.h5.

#### Run the Testing Script:
```bash
Run👇
python test.py
```
#### This will load the trained model and test it on sample messages.

## 🧑‍💻 Model Architecture
The LSTM model is designed as follows:
- **Embedding Layer:** Converts words into dense vectors
- **LSTM Layer:** Extracts sequential patterns in text
- **Dropout Layer:** Prevents overfitting
- **Dense Layers:** Fully connected layers for classification

## 📈 Training & Performance
The model was trained with the following parameters:
- **Optimizer:** Adam (`learning_rate=0.0003`)
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 32
- **Epochs:** 10

### 🔍 Accuracy & Loss Curves
The following curves illustrate the training accuracy and loss over epochs:

### 🔍 Accuracy & Loss Curves

The following curves illustrate the training accuracy and loss over epochs:

![Loss Curve](ham%20spam%20using%20rnn/loss_curve.png)

---

#### Thank You for Your Valuable Time!

I wanted to experiment with the LSTM network on textual data, and it was initially a bit challenging. Finding the right learning rate, optimizers, and the number of epochs took some time. However, after adjusting and fine-tuning the parameters, I achieved the results you see here.With the right values, you can definitely fine-tune the model to get the best results. Its a bit embarissing but Below, I will provide some initial validation and training loss curves:

![Loss Curve](ham%20spam%20using%20rnn/loss1.png)

![Loss Curve](ham%20spam%20using%20rnn/loss2.png)

![Loss Curve](ham%20spam%20using%20rnn/loss3.png)

Have a good day and feel positive


