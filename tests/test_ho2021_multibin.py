import pytest
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import os

from mfgpflow.data_loader import PowerSpecs
from mfgpflow.linear import MultiFidelityGPModel

@pytest.fixture
def generate_data():
    """Fixture to load the dataset once for all tests, relative to the project root."""
    test_dir = os.path.dirname(__file__)  # Get the directory where this test file is located
    data_folder = os.path.abspath(os.path.join(test_dir, "..", "data", "50_LR_3_HR"))  # Move up and into data/

    data = PowerSpecs()
    data.read_from_txt(folder=data_folder)  # Load data from the correct path
    return data

@pytest.fixture
def setup_model(generate_data):
    """Fixture to prepare multi-fidelity GP model."""
    data = generate_data
    X_LF, Y_LF = data.X_train_norm[0], data.Y_train_norm[0]
    X_HF, Y_HF = data.X_train_norm[1], data.Y_train_norm[1]
    
    X_test_HF, Y_test_HF = data.X_test_norm[0], data.Y_test[0]

    n_LF, n_HF = X_LF.shape[0], X_HF.shape[0]
    output_dim = Y_LF.shape[1]

    X_LF_aug = np.hstack([X_LF, np.zeros((n_LF, 1))])
    X_HF_aug = np.hstack([X_HF, np.ones((n_HF, 1))])
    X_train = np.vstack([X_LF_aug, X_HF_aug])
    Y_train = np.vstack([Y_LF, Y_HF])

    kernel_L = gpflow.kernels.RBF(lengthscales=np.ones(5), variance=1.0)
    kernel_delta = gpflow.kernels.RBF(lengthscales=np.ones(5), variance=1.0)

    mf_gp = MultiFidelityGPModel(X_train, Y_train, kernel_L, kernel_delta)
    mf_gp.optimize(max_iters=100, use_adam=True, learning_rate=0.1, unfix_noise_after=50)

    return data, mf_gp, X_test_HF, Y_test_HF

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_training_spectra(generate_data, save_dir):
    """Test function to visualize training spectra and save plots."""
    os.makedirs(save_dir, exist_ok=True)
    data = generate_data

    plt.figure()
    for i, y_train in enumerate(data.Y_train[0]):
        plt.loglog(10**data.kf, 10**y_train, color="C0", ls='--', alpha=0.3, label="low-fidelity" if i == 0 else "")
    for i, y_train in enumerate(data.Y_train[1]):
        plt.loglog(10**data.kf, 10**y_train, color="C1", label="high-fidelity" if i == 0 else "")
    
    plt.legend()
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$P_{k}$")
    plt.savefig(os.path.join(save_dir, "training_spectra.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_loss_history(setup_model, save_dir):
    """Test function to plot and save model training loss history."""
    os.makedirs(save_dir, exist_ok=True)
    _, mf_gp, _, _ = setup_model

    plt.figure()
    plt.plot(mf_gp.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_predictions(setup_model, save_dir):
    """Test function to visualize predictions vs. true values."""
    os.makedirs(save_dir, exist_ok=True)
    data, mf_gp, X_test_HF, Y_test_HF = setup_model

    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)

    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(np.arange(10), mean_pred[:, i], label="Predicted", alpha=0.7)
        plt.scatter(np.arange(10), Y_test_HF[:, i], label="True", alpha=0.5, marker="x")
        plt.fill_between(np.arange(10), 
                         (mean_pred[:, i] - 1.96 * np.sqrt(var_pred[:, i])).numpy(),
                         (mean_pred[:, i] + 1.96 * np.sqrt(var_pred[:, i])).numpy(),
                         color="lightgray", alpha=0.5, label="95% CI")
        plt.title(f"Output {i + 1}")
        plt.xlabel("Test Samples")
        plt.ylabel("Predicted Value")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_residuals(setup_model, save_dir):
    """Test function to plot residuals of predictions."""
    os.makedirs(save_dir, exist_ok=True)
    data, mf_gp, X_test_HF, Y_test_HF = setup_model

    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, var_pred = mf_gp.predict_f(X_test_HF_aug)
    residuals = mean_pred.numpy() - Y_test_HF

    plt.figure(figsize=(8, 5))
    plt.boxplot(residuals, showfliers=False)
    plt.axhline(0, linestyle="--", color="red", label="Zero Residual")
    plt.title("Residuals Across Outputs")
    plt.xlabel("Output Dimension")
    plt.ylabel("Residual")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "residuals.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_pred_vs_true(setup_model, save_dir):
    """Test function for predicted vs. true plot across k."""
    os.makedirs(save_dir, exist_ok=True)
    data, mf_gp, X_test_HF, Y_test_HF = setup_model

    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, _ = mf_gp.predict_f(X_test_HF_aug)

    plt.figure()
    plt.semilogx(10**data.kf, (10**mean_pred / 10**Y_test_HF).numpy().T)
    plt.ylim(0.9, 1.1)
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$P_{k,\mathrm{pred}}/P_{k,\mathrm{true}}$")
    plt.savefig(os.path.join(save_dir, "pred_vs_true.png"))
    plt.show()

@pytest.mark.parametrize("save_dir", ["test_plots"])
def test_plot_absolute_error(setup_model, save_dir):
    """Test function for absolute error plot."""
    os.makedirs(save_dir, exist_ok=True)
    data, mf_gp, X_test_HF, Y_test_HF = setup_model

    X_test_HF_aug = np.hstack([X_test_HF, np.ones((10, 1))])
    mean_pred, _ = mf_gp.predict_f(X_test_HF_aug)

    plt.figure()
    plt.semilogx(10**data.kf, np.abs((10**mean_pred / 10**Y_test_HF - 1).numpy()).mean(axis=0))
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$P_{k,\mathrm{pred}}-P_{k,\mathrm{true}}$")
    plt.savefig(os.path.join(save_dir, "absolute_error.png"))
    plt.show()

