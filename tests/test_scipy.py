import unittest
import numpy as np
import gpflow
import tensorflow as tf
from mfgpflow.linear import MultiFidelityGPModel, LinearMultiFidelityKernel

class TestMultiFidelityGP(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.X_L = np.random.uniform(-3, 3, (10, 1))
        self.Y_L = np.sin(self.X_L) + 0.1 * np.random.randn(10, 1)
        self.X_H = np.random.uniform(-3, 3, (5, 1))
        self.Y_H = 1.2 * np.sin(self.X_H) + 0.05 * np.random.randn(5, 1)

        self.X_L = np.hstack([self.X_L, np.zeros((self.X_L.shape[0], 1))])
        self.X_H = np.hstack([self.X_H, np.ones((self.X_H.shape[0], 1))])
        
        self.X = np.vstack([self.X_L, self.X_H])
        self.Y = np.vstack([self.Y_L, self.Y_H])
        
        self.kernel_L = gpflow.kernels.SquaredExponential()
        self.kernel_delta = gpflow.kernels.SquaredExponential()
        
        self.mf_gp = MultiFidelityGPModel(self.X, self.Y, self.kernel_L, self.kernel_delta)
    
    def test_scipy_optimizer(self):
        """Test if the scipy optimizer runs without errors and reduces loss."""
        initial_loss = self.mf_gp.training_loss().numpy()
        self.mf_gp.optimize(max_iters=500, use_adam=False)
        final_loss = self.mf_gp.training_loss().numpy()
        self.assertLess(final_loss, initial_loss, "Scipy optimizer should reduce loss.")
    
    def test_rho_learning_shape(self):
        """Test if rho is correctly learned with the right shape based on output dimensions."""
        num_output_dims = self.Y.shape[1]
        self.assertEqual(self.mf_gp.kernel.rho.shape[0], num_output_dims,
                         "rho should have as many elements as the output dimensions.")
    
    def test_kernel_psd(self):
        """Test if the kernel matrix is positive semi-definite."""
        K = self.mf_gp.kernel.K(self.X, self.X)
        eigvals = np.linalg.eigvalsh(K.numpy())
        self.assertTrue(np.all(eigvals >= -1e-6), "Kernel matrix should be positive semi-definite.")
    
    def test_prediction_shapes(self):
        """Test if predictions have the expected shape."""
        mean, var = self.mf_gp.predict_f(self.X)
        self.assertEqual(mean.shape, self.Y.shape, "Prediction mean shape should match Y shape.")
        self.assertEqual(var.shape, self.Y.shape, "Prediction variance shape should match Y shape.")

if __name__ == "__main__":
    unittest.main()
