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
        - f_L(x) is a Gaussian process modeling the low-fidelity function.
        - δ(x) is an independent GP modeling discrepancies.
        - ρ is a **learnable array** applied across **output dimensions**.

    The covariance matrix has a block structure:

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
        - num_output_dims: The number of independent outputs (size of `Y.shape[1]`).
    """

    def __init__(self, kernel_L, kernel_delta, num_output_dims):
        super().__init__()
        self.kernel_L = kernel_L  # Kernel for low-fidelity function
        self.kernel_delta = kernel_delta  # Kernel for discrepancy

        # Learnable rho for each output dimension
        self.rho = gpflow.Parameter(
            np.ones((num_output_dims, 1)), transform=positive()
        )

    def K(self, X, X2=None):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.
        """
        if X2 is None:
            X2 = X

        # Convert input to TensorFlow tensors
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # Extract fidelity indicators
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]
        mask2_L = tf.where(X2[:, -1] == 0)[:, 0]
        mask2_H = tf.where(X2[:, -1] == 1)[:, 0]

        # Extract LF and HF data
        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)
        X2_L = tf.gather(X2[:, :-1], mask2_L)
        X2_H = tf.gather(X2[:, :-1], mask2_H)

        # Ensure rho has shape `(output_dim, 1)` for broadcasting
        rho = tf.reshape(self.rho, [-1, 1])  # Shape: (output_dim, 1)

        # Compute covariance **for each output dimension**
        K_full = tf.zeros((X.shape[0], X2.shape[0]), dtype=tf.float64)

        for i in range(self.rho.shape[0]):  # Loop over output dimensions
            K_LL = self.kernel_L.K(X_L, X2_L)  # LF covariance
            K_LH = self.kernel_L.K(X_L, X2_H) * rho[i]  # LF-HF scaled covariance
            K_HL = self.kernel_L.K(X_H, X2_L) * rho[i]  # HF-LF scaled covariance
            K_HH = self.kernel_L.K(X_H, X2_H) * (rho[i] @ tf.transpose(rho[i])) + self.kernel_delta.K(X_H, X2_H)

            # Scatter updates to construct full covariance
            indices_LL = tf.stack(tf.meshgrid(mask_L, mask2_L, indexing="ij"), axis=-1)
            indices_LH = tf.stack(tf.meshgrid(mask_L, mask2_H, indexing="ij"), axis=-1)
            indices_HL = tf.stack(tf.meshgrid(mask_H, mask2_L, indexing="ij"), axis=-1)
            indices_HH = tf.stack(tf.meshgrid(mask_H, mask2_H, indexing="ij"), axis=-1)

            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LL, [-1, 2]), tf.reshape(K_LL, [-1]))
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LH, [-1, 2]), tf.reshape(K_LH, [-1]))
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HL, [-1, 2]), tf.reshape(K_HL, [-1]))
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HH, [-1, 2]), tf.reshape(K_HH, [-1]))

        return K_full

    def K_diag(self, X):
        """
        Computes the diagonal elements of the covariance matrix.
        """
        X = tf.convert_to_tensor(X, dtype=tf.float64)

        # Extract LF and HF indices
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]

        X_L = tf.gather(X[:, :-1], mask_L)
        X_H = tf.gather(X[:, :-1], mask_H)

        # Compute diagonal covariance elements
        K_diag_L = self.kernel_L.K_diag(X_L)
        K_diag_H = self.kernel_L.K_diag(X_H) * (self.rho ** 2) + self.kernel_delta.K_diag(X_H)

        # Construct full diagonal vector
        K_diag_full = tf.zeros((X.shape[0],), dtype=tf.float64)

        # Place diagonal values at correct indices
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_L, [-1, 1]), tf.reshape(K_diag_L, [-1]))
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_H, [-1, 1]), tf.reshape(K_diag_H, [-1]))

        return K_diag_full

class MultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for multi-fidelity learning.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta):
        num_output_dims = Y.shape[1]  # Number of output dimensions
        self.kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_output_dims)
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-3)
        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)
        set_trainable(self.likelihood.variance, False)

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        """
        Optimizes the model while using the noise fixing trick.
        """
        if use_adam:
            optimizer = tf.optimizers.Adam(learning_rate)

            @tf.function
            def optimization_step():
                with tf.GradientTape() as tape:
                    loss = self.training_loss()
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

            print("Optimizing with Adam...")
            for i in range(max_iters):
                optimization_step()

                if i == unfix_noise_after:
                    print(f"Unfixing noise at iteration {i}")
                    set_trainable(self.likelihood.variance, True)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {self.training_loss().numpy()}")
        else:
            print("Optimizing with L-BFGS (Scipy)...")
            scipy_optimizer = gpflow.optimizers.Scipy()
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})
            set_trainable(self.likelihood.variance, True)
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})