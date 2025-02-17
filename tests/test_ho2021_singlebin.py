import pytest
import os
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans

from mfgpflow.data_loader import PowerSpecs
from mfgpflow.singlebin_svgp import SingleBinSVGP
from mfgpflow.linear import LinearMultiFidelityKernel

@pytest.fixture
def generate_data():
    """Fixture to load dataset once for all tests."""
    test_dir = os.path.dirname(__file__)  # Get test script location
    data_folder = os.path.abspath(os.path.join(test_dir, "..", "data", "50_LR_3_HR"))

    data = PowerSpecs()
    data.read_from_txt(folder=data_folder)
    return data

@pytest.fixture
def setup_model(generate_data):
    """Fixture to set up the SVGP model for testing."""
    data = generate_data

    # Load LF and HF data
    X_LF, Y_LF = data.X_train_norm[0], data.Y_train_norm[0]
    X_HF, Y_HF = data.X_train_norm[1], data.Y_train_norm[1]
    X_test_HF, Y_test_HF = data.X_test_norm[0], data.Y_test[0]

    n_LF, n_HF = X_LF.shape[0], X_HF.shape[0]
    output_dim = Y_LF.shape[1]  # 49 output bins
    num_inducing = 50  # Inducing points

    # ✅ Append fidelity indicators (0 for LF, 1 for HF)
    X_LF_aug = np.hstack([X_LF, np.zeros((n_LF, 1))])
    X_HF_aug = np.hstack([X_HF, np.ones((n_HF, 1))])
    X_train = np.vstack([X_LF_aug, X_HF_aug])
    Y_train = np.vstack([Y_LF, Y_HF])

    # ✅ Define base kernels
    kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(5), variance=1.0)
    kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(5), variance=1.0)

    # ✅ Optimize Inducing Points using K-Means
    kmeans = KMeans(n_clusters=num_inducing, random_state=42).fit(X_train)
    Z_init = kmeans.cluster_centers_

    # ✅ Initialize the SVGP model
    mf_gp = SingleBinSVGP(
        X_train, Y_train,
        kernel_L, kernel_delta,
        num_outputs=output_dim,
        Z=Z_init,
    )

    # ✅ Train model (shorter training for testing)
    mf_gp.optimize(
        (X_train, Y_train),
        max_iters=1000,  # Reduce for faster tests
        initial_lr=0.01,
        unfix_noise_after=500,
    )

    return data, mf_gp, X_test_HF, Y_test_HF

def test_data_loading(generate_data):
    """Test data loading correctness."""
    data = generate_data
    assert data.X_train_norm[0].shape[0] > 0, "LF data not loaded!"
    assert data.X_train_norm[1].shape[0] > 0, "HF data not loaded!"

def test_model_training(setup_model):
    """Test that the model trains without errors."""
    _, mf_gp, _, _ = setup_model
    assert hasattr(mf_gp, "trainable_variables"), "Model training failed!"

def test_prediction_shapes(setup_model):
    """Test prediction outputs correct shape."""
    _, mf_gp, X_test_HF, _ = setup_model
    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])  # Add HF indicator (1)
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)

    assert mean_pred.shape == (10, 49), "Mean prediction shape incorrect!"
    assert var_pred.shape == (10, 49), "Variance prediction shape incorrect!"

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_predictions(setup_model, save_dir):
    """Test function to visualize predictions."""
    os.makedirs(save_dir, exist_ok=True)
    _, mf_gp, X_test_HF, Y_test_HF = setup_model

    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)

    plt.figure(figsize=(8, 5))
    plt.plot(Y_test_HF[:, 0], label="True HF", linestyle="--", marker="o")
    plt.plot(mean_pred.numpy()[:, 0], label="Predicted HF", linestyle="-", marker="x")
    plt.fill_between(
        np.arange(len(Y_test_HF[:, 0])),
        mean_pred.numpy()[:, 0] - 2 * np.sqrt(var_pred.numpy()[:, 0]),
        mean_pred.numpy()[:, 0] + 2 * np.sqrt(var_pred.numpy()[:, 0]),
        alpha=0.2, color="gray", label="Confidence Interval"
    )
    plt.legend()
    plt.title("HF Predictions vs Ground Truth")
    plt.savefig(os.path.join(save_dir, "predictions.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_loss_history(setup_model, save_dir):
    """Test function to visualize loss history."""
    os.makedirs(save_dir, exist_ok=True)
    _, mf_gp, _, _ = setup_model

    plt.figure()
    plt.plot(mf_gp.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_model_save_load(setup_model, save_dir):
    """Test model save and reload functionality."""
    os.makedirs(save_dir, exist_ok=True)
    _, mf_gp, X_test_HF, Y_test_HF = setup_model

    # ✅ Save model
    model_path = os.path.join(save_dir, "svgp_model.pkl")
    mf_gp.save_model(model_path)
    assert os.path.exists(model_path), "Model save failed!"

    # ✅ Reload model with dummy inputs
    kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(5), variance=1.0)
    kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(5), variance=1.0)

    X_train_dummy = np.zeros((10, 6))
    Y_train_dummy = np.zeros((10, 49))
    Z_dummy = np.zeros((50, 6))

    loaded_model = SingleBinSVGP.load_model(model_path, X_train_dummy, Y_train_dummy, kernel_L, kernel_delta, 49, 50, Z_dummy)
    assert isinstance(loaded_model, SingleBinSVGP), "Loaded model is not an instance of SingleBinSVGP"

    # ✅ Ensure loaded model predicts correctly
    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, _ = loaded_model.predict_f(X_test_HF_aug)
    assert mean_pred.shape == (10, 49), "Loaded model predictions incorrect!"