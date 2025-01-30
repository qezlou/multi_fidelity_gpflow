import unittest
import numpy as np
import gpflow
import tensorflow as tf
from mfgpflow.linear import MultiFidelityGPModel  # Ensure you import the correct module
from mfgpflow.linear import LinearMultiFidelityKernel

class TestMultiFidelityGPMultiOutput(unittest.TestCase):
    """
    Unit test for multi-output handling of rho in MultiFidelityGPModel.
    """

    def setUp(self):
        """
        Set up a simple multi-output dataset for testing.
        """
        np.random.seed(42)

        # Generate synthetic multi-output data
        D = 3  # Number of output dimensions
        X_L = np.random.uniform(-3, 3, (10, 1))
        Y_L = np.sin(X_L) + 0.1 * np.random.randn(10, D)  # Multi-output LF
        X_H = np.random.uniform(-3, 3, (5, 1))
        Y_H = 1.2 * np.sin(X_H) + 0.05 * np.random.randn(5, D)  # Multi-output HF

        # Append fidelity indicator
        X_L = np.hstack([X_L, np.zeros((X_L.shape[0], 1))])
        X_H = np.hstack([X_H, np.ones((X_H.shape[0], 1))])

        # Merge datasets
        self.X = np.vstack([X_L, X_H])
        self.Y = np.vstack([Y_L, Y_H])

        # Define kernels
        kernel_L = gpflow.kernels.SquaredExponential()
        kernel_delta = gpflow.kernels.SquaredExponential()

        # Initialize multi-fidelity GP model
        self.mf_gp = MultiFidelityGPModel(self.X, self.Y, kernel_L, kernel_delta)

    def test_rho_shape(self):
        """
        Test if rho is correctly initialized and has the right shape.
        """
        rho = self.mf_gp.kernel.rho.numpy()
        expected_shape = (1, self.Y.shape[1])  # Shape should match the output dimensions

        print(f"\n🔍 Initial rho shape: {rho.shape} (Expected: {expected_shape})")
        self.assertEqual(rho.shape, expected_shape, "rho shape mismatch!")

    def test_rho_updates_after_optimization(self):
        """
        Test if rho updates after training the model.
        """
        initial_rho = self.mf_gp.kernel.rho.numpy().copy()
        print(f"\n🔍 Initial rho values: {initial_rho}")

        # Optimize the model
        self.mf_gp.optimize(max_iters=500, use_adam=False)

        updated_rho = self.mf_gp.kernel.rho.numpy()
        print(f"\n🔍 Updated rho values: {updated_rho}")

        # Ensure rho has changed
        self.assertFalse(np.allclose(initial_rho, updated_rho), "rho did not change after optimization!")

if __name__ == "__main__":
    unittest.main()