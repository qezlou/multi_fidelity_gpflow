import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from gpflow.utilities import set_trainable
from linear import MultiFidelityGPModel  # Assuming the kernel and model are defined in linear.py

def forrester(x, sd=0):
    x = x.reshape((len(x), 1))
    fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    noise = np.random.normal(0, sd, x.shape) if sd > 0 else np.zeros_like(x)
    return fval + noise

def forrester_low(x, sd=0):
    return 0.5 * forrester(x, 0) + 10 * (x - 0.5) + 5 + np.random.randn(*x.shape) * sd

# Generate Training Data
np.random.seed(42)
x_train_l = np.random.rand(60, 1)
x_train_h = np.random.permutation(x_train_l)[:20]
y_train_l = forrester_low(x_train_l, sd=0.05)
y_train_h = forrester(x_train_h, sd=0.02)

X_L = np.hstack([x_train_l, np.zeros_like(x_train_l)])
X_H = np.hstack([x_train_h, np.ones_like(x_train_h)])
X = np.vstack([X_L, X_H])
Y = np.vstack([y_train_l, y_train_h])

# Generate Test Data
x_plot = np.linspace(0, 1, 200)[:, None]
X_L_plot = np.hstack([x_plot, np.zeros_like(x_plot)])
X_H_plot = np.hstack([x_plot, np.ones_like(x_plot)])

def plot_predictions(model, X_plot, label):
    mean, var = model.predict_f(X_plot)
    std = np.sqrt(var)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, forrester(x_plot), 'k--', label='True HF Function')
    plt.plot(x_plot, forrester_low(x_plot), 'b--', label='True LF Function')
    plt.plot(X_plot[:, 0], mean, 'r', label=f'Predicted {label}')
    plt.fill_between(X_plot[:, 0], mean[:, 0] - 2 * std[:, 0], mean[:, 0] + 2 * std[:, 0], alpha=0.2, color='red')
    plt.scatter(X[:, 0], Y, color='black', marker='x', label='Training Data')
    plt.legend()
    plt.title(f'{label} Predictions with Confidence Intervals')
    plt.show()

def plot_residuals(model, X_plot, true_func, label):
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

def validate_model(model):
    """ Runs simple checks on model consistency. """
    assert model.kernel.rho.shape == (Y.shape[1], 1), "rho shape incorrect"
    K = model.kernel.K(X, X).numpy()
    assert np.all(np.linalg.eigvals(K) >= -1e-8), "Covariance matrix is not PSD"
    print("✅ Model passes validation tests!")

# Define and Train Model
kernel_L = gpflow.kernels.SquaredExponential()
kernel_delta = gpflow.kernels.SquaredExponential()
mf_gp = MultiFidelityGPModel(X, Y, kernel_L, kernel_delta)
mf_gp.optimize(max_iters=1000, use_adam=True)

# Run tests and plots
validate_model(mf_gp)
plot_predictions(mf_gp, X_L_plot, "Low-Fidelity")
plot_predictions(mf_gp, X_H_plot, "High-Fidelity")
plot_residuals(mf_gp, X_H_plot, forrester, "High-Fidelity")
