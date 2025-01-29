import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

class LinearMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Linear Multi-Fidelity Kernel (Kennedy & O’Hagan, 2000).

    This kernel models a high-fidelity function as:

        f_H(x) = ρ(x) * f_L(x) + δ(x)

    where:
        - f_L(x) is a Gaussian process modeling the low-fidelity function.
        - δ(x) is an independent GP modeling discrepancies.
        - ρ(x) is a learnable scaling function with independent values for each HF data point.

    The covariance matrix is structured as:

        K =
        [  K_LL   K_LH  ]
        [  K_HL   K_HH  ]

    where:
        - K_LL = Covariance matrix for low-fidelity points.
        - K_LH = K_HL^T = Scaled cross-covariance.
        - K_HH = Scaled LF + discrepancy covariance.

    Parameters:
        - kernel_L: GP kernel for the low-fidelity function.
        - kernel_delta: GP kernel for the discrepancy.
        - num_HF_points: Number of high-fidelity data points (to define ρ as a vector).
    """

    def __init__(self, kernel_L, kernel_delta, num_HF_points):
        super().__init__()
        self.kernel_L = kernel_L  # Kernel for the low-fidelity function
        self.kernel_delta = kernel_delta  # Kernel for the discrepancy function
        self.rho = gpflow.Parameter(np.ones(num_HF_points), transform=positive())  # Learnable rho(x) as an array

    def K(self, X, X2=None):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.
        """

        # Separate LF and HF data (last column is fidelity indicator)
        X_L = X[X[:, -1] == 0][:, :-1]  # Low-fidelity inputs
        X_H = X[X[:, -1] == 1][:, :-1]  # High-fidelity inputs
        if X2 is not None:
            X2_L = X2[X2[:, -1] == 0][:, :-1]
            X2_H = X2[X2[:, -1] == 1][:, :-1]
        else:
            X2_L, X2_H = X_L, X_H

        # Compute covariance components
        K_LL = self.kernel_L.K(X_L, X2_L)  # LF-LF covariance
        K_LH = self.kernel_L.K(X_L, X2_H) * self.rho[:, None]  # Scaled LF-HF covariance
        K_HL = self.kernel_L.K(X_H, X2_L) * self.rho[None, :]  # Transposed scaling
        K_HH = self.kernel_L.K(X_H, X2_H) * (self.rho[:, None] @ self.rho[None, :]) + self.kernel_delta.K(X_H, X2_H)

        # Assemble the block matrix
        return tf.concat([
            tf.concat([K_LL, K_LH], axis=1),
            tf.concat([K_HL, K_HH], axis=1)
        ], axis=0)

    def K_diag(self, X):
        """
        Computes the diagonal elements of the covariance matrix.
        """
        X_L = X[X[:, -1] == 0][:, :-1]
        X_H = X[X[:, -1] == 1][:, :-1]

        # Compute diagonal covariance
        K_diag_L = self.kernel_L.K_diag(X_L)
        K_diag_H = self.kernel_L.K_diag(X_H) * self.rho**2 + self.kernel_delta.K_diag(X_H)

        return tf.concat([K_diag_L, K_diag_H], axis=0)


class MultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for multi-fidelity learning.

    Uses a single GPFlow optimizer to train both LF and HF models jointly.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta):
        """
        :param X: Training inputs (with fidelity indicator).
        :param Y: Training targets.
        :param kernel_L: Kernel for the low-fidelity function.
        :param kernel_delta: Kernel for the discrepancy function.
        """
        num_HF = np.sum(X[:, -1] == 1)  # Number of high-fidelity data points
        self.kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_HF)
        likelihood = gpflow.likelihoods.Gaussian()
        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)

    def optimize(self, max_iters=1000):
        """
        Trains the multi-fidelity model using a single optimizer.
        """
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    X_L = np.random.uniform(-3, 3, (10, 1))
    Y_L = np.sin(X_L) + 0.1 * np.random.randn(10, 1)
    X_H = np.random.uniform(-3, 3, (5, 1))
    Y_H = 1.2 * np.sin(X_H) + 0.05 * np.random.randn(5, 1)

    # Append fidelity indicator
    X_L = np.hstack([X_L, np.zeros((X_L.shape[0], 1))])
    X_H = np.hstack([X_H, np.ones((X_H.shape[0], 1))])

    # Merge datasets
    X = np.vstack([X_L, X_H])
    Y = np.vstack([Y_L, Y_H])

    # Define kernels
    kernel_L = gpflow.kernels.SquaredExponential()
    kernel_delta = gpflow.kernels.SquaredExponential()

    # Train model
    mf_gp = MultiFidelityGPModel(X, Y, kernel_L, kernel_delta)
    mf_gp.optimize()

    # Print learned rho values
    print("Learned rho values:", mf_gp.kernel.rho.numpy())