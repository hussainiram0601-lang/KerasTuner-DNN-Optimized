# AutoTuner-Diabetes-Classifier 🩺🤖

A deep learning classification pipeline built with **TensorFlow/Keras** and optimized using **KerasTuner** to predict the onset of diabetes based on diagnostic measurements.

## 📊 Project Overview
This project preprocesses clinical data, standardizes features, builds a Sequential Neural Network, and automates hyperparameter optimization using `RandomSearch`. It explores various hyperparameter combinations—including network depth (number of layers), layer units, activation functions, dropout rates, and optimizers—to maximize validation accuracy.

## ⚙️ Tech Stack & Libraries
* **Python** 
* **TensorFlow / Keras** (Neural Network modeling & architecture)
* **KerasTuner** (`RandomSearch` for automated hyperparameter tuning)
* **Scikit-Learn** (`StandardScaler`, `train_test_split`)
* **Pandas & NumPy** (Data preprocessing & exploration)

## 🚀 Key Steps in the Pipeline
1. **Data Loading & EDA:** Inspected feature distributions and correlation matrices on the diabetes dataset.
2. **Data Preprocessing:** Standardized features using `StandardScaler` to ensure zero mean and unit variance.
3. **Hyperparameter Tuning:** Defined a dynamic model builder function and used `KerasTuner`'s Random Search to search over optimal layers, units, dropout rates, and optimizers.
4. **Model Training & Evaluation:** Retrained the best-performing model configuration over multiple epochs and evaluated its generalization capability on test validation data.

## 📂 Dataset Source
Pima Indians Diabetes Dataset (commonly hosted via Kaggle / standard ML repositories).
