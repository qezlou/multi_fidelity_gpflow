import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive, set_trainable


class LinearMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Linear Multi-Fidelity Kernel (Kennedy & O’Hagan, 2000).
    
    This kernel models the high-fidelity function as:

        f_H(x) = ρ * f_L(x) + δ(x)

    where:
    - f_L(x) is the Gaussian process modeling the low-fidelity function.
    - δ(x) is an independent GP modeling discrepancies.
    - ρ is a learnable scaling parameter for each output dimension of Y.

    The covariance matrix has a block structure:

        K =
        [  K_LL   K_LH  ]
        [  K_HL   K_HH  ]

    Parameters:
    - kernel_L: GP kernel for the low-fidelity function.
    - kernel_delta: GP kernel for the discrepancy function.
    - output_dim: Number of output dimensions (defines learnable `rho`).
    """

    def __init__(self, kernel_L, kernel_delta, output_dim):
        super().__init__()
        self.kernel_L = kernel_L
        self.kernel_delta = kernel_delta

        # Learnable rho for each output dimension
        self.rho = gpflow.Parameter(
            np.ones(output_dim), transform=positive()
        )

    def K(self, X, X2=None):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.
        """
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # Extract LF and HF indices
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]
        mask2_L = tf.where(X2[:, -1] == 0)[:, 0]
        mask2_H = tf.where(X2[:, -1] == 1)[:, 0]

        # Extract input points for each fidelity
        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)
        X2_L = tf.gather(X2[:, :-1], mask2_L)
        X2_H = tf.gather(X2[:, :-1], mask2_H)

        # Ensure rho is properly broadcasted across dimensions
        rho_diag = tf.linalg.diag(self.rho)  # Shape (output_dim, output_dim)

        # Compute covariance components
        K_LL = self.kernel_L.K(X_L, X2_L)  # LF covariance
        K_LH = self.kernel_L.K(X_L, X2_H) @ rho_diag  # LF-HF covariance
        K_HL = rho_diag @ self.kernel_L.K(X_H, X2_L)  # HF-LF covariance
        K_HH = rho_diag @ self.kernel_L.K(X_H, X2_H) @ rho_diag + self.kernel_delta.K(X_H, X2_H)

        # Construct full covariance matrix
        K_full = tf.zeros((tf.shape(X)[0], tf.shape(X2)[0]), dtype=tf.float64)
        K_full = tf.tensor_scatter_nd_update(K_full, tf.stack([mask_L[:, None], mask2_L[None, :]], axis=-1), K_LL)
        K_full = tf.tensor_scatter_nd_update(K_full, tf.stack([mask_L[:, None], mask2_H[None, :]], axis=-1), K_LH)
        K_full = tf.tensor_scatter_nd_update(K_full, tf.stack([mask_H[:, None], mask2_L[None, :]], axis=-1), K_HL)
        K_full = tf.tensor_scatter_nd_update(K_full, tf.stack([mask_H[:, None], mask2_H[None, :]], axis=-1), K_HH)

        return K_full

    def K_diag(self, X):
        """
        Computes the diagonal elements of the covariance matrix.
        """
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]
        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)

        K_diag_L = self.kernel_L.K_diag(X_L)
        K_diag_H = self.kernel_L.K_diag(X_H) * tf.square(self.rho) + self.kernel_delta.K_diag(X_H)

        K_diag_full = tf.zeros((tf.shape(X)[0],), dtype=tf.float64)
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, mask_L[:, None], K_diag_L)
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, mask_H[:, None], K_diag_H)

        return K_diag_full


class MultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for multi-fidelity learning.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta):
        output_dim = Y.shape[1]  # Define rho based on output dimensions
        self.kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, output_dim)
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-3)
        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)
        set_trainable(self.likelihood.variance, False)

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        if use_adam:
            optimizer = tf.optimizers.Adam(learning_rate)
            for i in range(max_iters):
                with tf.GradientTape() as tape:
                    loss = self.training_loss()
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if i == unfix_noise_after:
                    set_trainable(self.likelihood.variance, True)
        else:
            scipy_optimizer = gpflow.optimizers.Scipy()
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})
            set_trainable(self.likelihood.variance, True)
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})
