import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from gpflow.utilities import set_trainable


import gpflow
import numpy as np
import tensorflow as tf


class LinearMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Linear Multi-Fidelity Kernel (Kennedy & O’Hagan, 2000).

    This kernel models the high-fidelity function as:

        f_H(x) = ρ(x) * f_L(x) + δ(x)

    where:
    - f_L(x) is the Gaussian process modeling the low-fidelity function.
    - δ(x) is an independent GP modeling the discrepancies.
    - ρ(x) is a learnable scaling function with independent values for each high-fidelity data point.

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
    - kernel_delta: GP kernel for the discrepancy function.
    - num_HF_points: Number of high-fidelity data points (to define ρ as a vector).
    """

    def __init__(self, kernel_L, kernel_delta, num_HF_points):
        super().__init__()
        self.kernel_L = kernel_L  # Kernel for the low-fidelity function
        self.kernel_delta = kernel_delta  # Kernel for the discrepancy function
        self.rho = gpflow.Parameter(
            np.ones(num_HF_points), transform=gpflow.utilities.positive()
        )  # Learnable rho(x)

    def K(self, X, X2=None):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.
        Includes debug printing for shape mismatches.
        """

        # Convert input to TensorFlow tensors
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        print(f"Inside K function")
        print(f"  X shape: {X.shape}, X2 shape: {X2.shape}")

        # Separate LF and HF data using boolean masks
        X_L = tf.boolean_mask(X, X[:, -1] == 0)[:, :-1]  # Low-fidelity inputs
        X_H = tf.boolean_mask(X, X[:, -1] == 1)[:, :-1]  # High-fidelity inputs

        X2_L = tf.boolean_mask(X2, X2[:, -1] == 0)[:, :-1]
        X2_H = tf.boolean_mask(X2, X2[:, -1] == 1)[:, :-1]

        print(f"  X_L shape: {X_L.shape}, X_H shape: {X_H.shape}")
        print(f"  X2_L shape: {X2_L.shape}, X2_H shape: {X2_H.shape}")

        # Ensure rho is a column vector of shape [HF_points, 1]
        rho_col = tf.reshape(self.rho, (-1, 1))  # Shape: [HF_points, 1]

        print(f"  rho shape before reshape: {self.rho.shape}")
        print(f"  rho_col shape after reshape: {rho_col.shape}")

        # Compute covariance components
        K_LL = self.kernel_L.K(X_L, X2_L)
        K_LH = self.kernel_L.K(X_L, X2_H) * tf.reshape(self.rho, (-1, 1))
        K_HL = self.kernel_L.K(X_H, X2_L) * tf.reshape(self.rho, (1, -1))

        # Fix rho outer product to ensure correct shape [HF_points, HF_points]
        rho_outer = rho_col @ tf.transpose(rho_col)
        print(f"  rho_outer shape: {rho_outer.shape}")

        K_HH = self.kernel_L.K(X_H, X2_H) * rho_outer + self.kernel_delta.K(X_H, X2_H)

        print(f"  K_LL shape: {K_LL.shape}, K_LH shape: {K_LH.shape}")
        print(f"  K_HL shape: {K_HL.shape}, K_HH shape: {K_HH.shape}")

        # Initialize full covariance matrix using tf.zeros() instead of tf.Variable()
        K_full = tf.zeros((X.shape[0], X2.shape[0]), dtype=tf.float64)

        # Extract index positions for placing the computed covariance submatrices
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]
        mask2_L = tf.where(X2[:, -1] == 0)[:, 0]
        mask2_H = tf.where(X2[:, -1] == 1)[:, 0]

        print(f"  mask_L shape: {mask_L.shape}, mask_H shape: {mask_H.shape}")
        print(f"  mask2_L shape: {mask2_L.shape}, mask2_H shape: {mask2_H.shape}")

        # Ensure correct dimensions for indices
        indices_LL = tf.stack(tf.meshgrid(mask_L, mask2_L, indexing="ij"), axis=-1)
        indices_LH = tf.stack(tf.meshgrid(mask_L, mask2_H, indexing="ij"), axis=-1)
        indices_HL = tf.stack(tf.meshgrid(mask_H, mask2_L, indexing="ij"), axis=-1)
        indices_HH = tf.stack(tf.meshgrid(mask_H, mask2_H, indexing="ij"), axis=-1)

        print(f"  indices_LL shape: {indices_LL.shape}")
        print(f"  indices_HH shape: {indices_HH.shape}")

        # Apply tensor updates to construct the full covariance matrix
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LL, (-1, 2)), tf.reshape(K_LL, (-1,)))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LH, (-1, 2)), tf.reshape(K_LH, (-1,)))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HL, (-1, 2)), tf.reshape(K_HL, (-1,)))
        K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HH, (-1, 2)), tf.reshape(K_HH, (-1,)))

        print(f"  Final K_full shape: {K_full.shape}")

        return K_full

    def K_diag(self, X):
        """
        Computes the diagonal elements of the covariance matrix.
        """
        X = tf.convert_to_tensor(X, dtype=tf.float64)

        # Extract LF and HF indices
        X_L = tf.boolean_mask(X, X[:, -1] == 0)[:, :-1]
        X_H = tf.boolean_mask(X, X[:, -1] == 1)[:, :-1]

        # Compute diagonal covariance elements
        K_diag_L = self.kernel_L.K_diag(X_L)
        K_diag_H = self.kernel_L.K_diag(X_H) * self.rho**2 + self.kernel_delta.K_diag(X_H)

        # Construct full diagonal vector using tf.zeros() instead of tf.Variable()
        K_diag_full = tf.zeros((X.shape[0],), dtype=tf.float64)
        mask_L = tf.where(X[:, -1] == 0)[:, 0]
        mask_H = tf.where(X[:, -1] == 1)[:, 0]

        # Place diagonal values at correct indices
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_L, (-1, 1)), tf.reshape(K_diag_L, (-1,)))
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(mask_H, (-1, 1)), tf.reshape(K_diag_H, (-1,)))

        print(f"  Final K_diag shape: {K_diag_full.shape}")

        return K_diag_full

class MultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for multi-fidelity learning.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta):
        num_HF = np.sum(X[:, -1] == 1)  # Number of HF points
        self.kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_HF)

        # Use a slightly larger noise variance to avoid numerical instability
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-3)  
        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)

        # Fix noise AFTER initializing the model
        self.likelihood.variance.assign(1e-3)
        # Fix noise AFTER initializing the model using gpflow.set_trainable
        set_trainable(self.likelihood.variance, False)  # ✅ Correct way

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        """
        Optimizes the model while using the noise fixing trick.

        :param max_iters: Total optimization steps.
        :param learning_rate: Adam optimizer learning rate.
        :param use_adam: If True, uses Adam; otherwise, uses L-BFGS.
        :param unfix_noise_after: Iteration to unfix the noise.
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
                    set_trainable(self.likelihood.variance, True)  # ✅ Correct way

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {self.training_loss().numpy()}")
        else:
            print("Optimizing with L-BFGS (Scipy)...")
            scipy_optimizer = gpflow.optimizers.Scipy()
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})

            # After L-BFGS, unfix the noise and optimize again
            print("Unfixing noise after first optimization...")
            set_trainable(self.likelihood.variance, True)  # ✅ Correct way
            scipy_optimizer.minimize(self.training_loss, self.trainable_variables, options={"maxiter": max_iters})

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