# Power Plant Energy Output Prediction

A deep learning project that predicts the net hourly electrical energy output of a combined cycle power plant using a custom Artificial Neural Network (ANN) built with PyTorch.

## 🎯 Objective

The goal of this project is to map ambient environmental variables to continuous energy output using regression techniques. By utilizing a deep learning approach, the model learns non-linear relationships between the operating conditions of the plant and its resulting power generation.

## 🗄️ Dataset

The dataset contains 9,568 data points collected from a Combined Cycle Power Plant over 6 years, when the plant was set to work with full load.

**Input Features:**
* **AT:** Ambient Temperature (°C)
* **V:** Exhaust Vacuum (cm Hg)
* **AP:** Ambient Pressure (mbar)
* **RH:** Relative Humidity (%)

**Target Variable:**
* **PE:** Net hourly electrical energy output (MW)

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing & Metrics:** Scikit-Learn (`StandardScaler`, `train_test_split`, `r2_score`)
* **Visualization:** Matplotlib

## 🧠 Model Architecture

The model is a fully connected feedforward Artificial Neural Network constructed using PyTorch's `nn.Module`.

* **Input Layer:** 4 Features
* **Hidden Layer 1:** 6 Neurons (ReLU Activation)
* **Hidden Layer 2:** 6 Neurons (ReLU Activation)
* **Output Layer:** 1 Neuron (Linear output for regression)

The network was trained over 100 epochs using the **Adam Optimizer** and **Mean Squared Error (MSE)** as the loss function.

## 📊 Results

The model successfully generalized the data with high predictive accuracy, demonstrating a strong correlation between the predicted and actual energy output.

* **Testing Mean Squared Error (MSE):** ~19.62
* **R² Score:** 0.9314

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aamir-e/power-plant-ann-prediction.git
   cd power-plant-ann-prediction
