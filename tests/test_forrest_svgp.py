import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from gpflow.utilities import set_trainable
from mfgpflow.linear_svgp import LatentMFCoregionalizationSVGP  # ✅ Import the new SVGP model

# Include the import path from the previous directory
import sys
sys.path.append("..")

# 🔹 1️⃣ Define True Functions (High & Low Fidelity)
def forrester(x, sd=0):
    """ High-fidelity function with optional noise. """
    x = x.reshape((len(x), 1))
    fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    noise = np.random.normal(0, sd, x.shape) if sd > 0 else np.zeros_like(x)
    return fval + noise

def forrester_low(x, sd=0):
    """ Low-fidelity approximation of Forrester function. """
    return 0.5 * forrester(x, 0) + 10 * (x - 0.5) + 5 + np.random.randn(*x.shape) * sd

# 🔹 2️⃣ Generate Training Data
np.random.seed(42)
x_train_l = np.random.rand(60, 1)  # 60 LF points
x_train_h = np.random.permutation(x_train_l)[:20]  # 20 HF points
y_train_l = forrester_low(x_train_l, sd=0.05)
y_train_h = forrester(x_train_h, sd=0.02)

# Append fidelity indicators (0 for LF, 1 for HF)
X_L = np.hstack([x_train_l, np.zeros_like(x_train_l)])  # LF Inputs
X_H = np.hstack([x_train_h, np.ones_like(x_train_h)])   # HF Inputs
X_train = np.vstack([X_L, X_H])  # Combine (N=80, D=2)
Y_train = np.vstack([y_train_l, y_train_h])  # (80, P=1)

# 🔹 3️⃣ Generate Test Data
x_plot = np.linspace(0, 1, 200)[:, None]  # 200 test points
X_L_plot = np.hstack([x_plot, np.zeros_like(x_plot)])  # LF Test Inputs
X_H_plot = np.hstack([x_plot, np.ones_like(x_plot)])   # HF Test Inputs

# 🔹 4️⃣ Plotting Functions
def plot_predictions(model, X_plot, label):
    """ Plot GP Predictions with confidence intervals. """
    mean, var = model.predict_f(X_plot)
    std = np.sqrt(var)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, forrester(x_plot), 'k--', label='True HF Function')
    plt.plot(x_plot, forrester_low(x_plot), 'b--', label='True LF Function')
    plt.plot(X_plot[:, 0], mean, 'r', label=f'Predicted {label}')
    plt.fill_between(X_plot[:, 0], mean[:, 0] - 2 * std[:, 0], mean[:, 0] + 2 * std[:, 0], alpha=0.2, color='red')
    plt.scatter(X_train[:, 0], Y_train, color='black', marker='x', label='Training Data')
    plt.legend()
    plt.title(f'{label} Predictions with Confidence Intervals')
    plt.show()

def plot_residuals(model, X_plot, true_func, label):
    """ Plot residuals (error vs true function). """
    mean, _ = model.predict_f(X_plot)
    true_values = true_func(X_plot[:, :1])
    residuals = true_values - mean

    plt.figure(figsize=(8, 5))
    plt.scatter(X_plot[:, 0], residuals, color='red', label='Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'Residuals for {label}')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def validate_model(model, X):
    """ Runs simple checks on model consistency. """
    K = model.kernel.K(X, X).numpy()
    K = np.squeeze(K)  # ✅ Fix shape issue
    assert K.shape[0] == K.shape[1], "❌ K(X, X) is not square!"
    assert np.all(np.linalg.eigvals(K) >= -1e-8), "❌ Covariance matrix is not PSD!"
    assert model.W.shape == (Y_train.shape[1], model.num_latents), "❌ W shape mismatch!"
    print("✅ Model passes validation tests!")

# Define and Train Model
kernel_L = gpflow.kernels.SquaredExponential()
kernel_delta = gpflow.kernels.SquaredExponential()
Z = X_train[:100]  # 20 Inducing Points

mf_gp = LatentMFCoregionalizationSVGP(X_train, Y_train, kernel_L, kernel_delta, num_latents=1, num_outputs=Y_train.shape[1], Z=Z)
mf_gp.optimize((X_train, Y_train), max_iters=10000, initial_lr=0.005)

# 🔹 7️⃣ Run Validation & Generate Plots
validate_model(mf_gp, X_train)
plot_predictions(mf_gp, X_L_plot, "Low-Fidelity")
plot_predictions(mf_gp, X_H_plot, "High-Fidelity")
plot_residuals(mf_gp, X_H_plot, forrester, "High-Fidelity")
