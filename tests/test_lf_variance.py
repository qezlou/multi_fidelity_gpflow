import unittest
import numpy as np
import gpflow
import tensorflow as tf
from gpflow.utilities import set_trainable

from mfgpflow.linear import MultiFidelityGPModel

class TestMultiFidelityLFVariance(unittest.TestCase):
    """
    Test that LF variance is correctly learned and not overestimated.
    """

    @classmethod
    def setUpClass(cls):
        """
        Generate a simple multi-fidelity dataset for testing.
        """
        np.random.seed(42)

        # Define synthetic LF and HF functions
        def forrester(x):
            return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)

        def forrester_low(x):
            return 0.5 * forrester(x) + 10 * (x[:, [0]] - 0.5) + 5

        # Generate LF (60 points) and HF (20 points)
        cls.X_L = np.random.rand(60, 1)
        cls.X_H = np.random.permutation(cls.X_L)[:20]

        cls.Y_L = forrester_low(cls.X_L) + 0.05 * np.random.randn(60, 1)
        cls.Y_H = forrester(cls.X_H) + 0.01 * np.random.randn(20, 1)

        # Append fidelity indicators
        cls.X_L = np.hstack([cls.X_L, np.zeros_like(cls.X_L)])
        cls.X_H = np.hstack([cls.X_H, np.ones_like(cls.X_H)])

        # Merge datasets
        cls.X = np.vstack([cls.X_L, cls.X_H])
        cls.Y = np.vstack([cls.Y_L, cls.Y_H])

        # Define kernels
        kernel_L = gpflow.kernels.SquaredExponential()
        kernel_delta = gpflow.kernels.SquaredExponential()

        # Initialize model
        cls.mf_gp = MultiFidelityGPModel(cls.X, cls.Y, kernel_L, kernel_delta)

    def test_lf_variance_consistency(self):
        """
        Test that LF variance does not get overestimated after training.
        """

        # 🔹 Predict variance at LF points *before* training
        _, var_LF_prior = self.mf_gp.predict_f(self.X_L)

        # 🔹 Train the model
        self.mf_gp.optimize(max_iters=500, use_adam=False)

        # 🔹 Predict variance at LF points *after* training
        _, var_LF_post = self.mf_gp.predict_f(self.X_L)

        # Convert to numpy arrays for testing
        var_LF_prior = var_LF_prior.numpy().flatten()
        var_LF_post = var_LF_post.numpy().flatten()

        # ✅ 1️⃣ Ensure variance does not increase after training
        np.testing.assert_array_less(var_LF_post, var_LF_prior * 1.2, err_msg="LF variance increased too much!")

        # ✅ 2️⃣ Ensure LF variance is **not drastically larger** than HF
        _, var_HF_post = self.mf_gp.predict_f(self.X_H)
        var_HF_post = var_HF_post.numpy().flatten()
        np.testing.assert_array_less(var_LF_post, var_HF_post * 5, err_msg="LF variance too large relative to HF!")

if __name__ == "__main__":
    unittest.main()