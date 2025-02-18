import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import positive, set_trainable, add_likelihood_noise_cov, assert_params_false
import tensorflow_probability as tfp

class GraphMultiFidelityKernel(gpflow.kernels.Kernel):
    """
    Graph-structured Multi-Fidelity Kernel with learnable cross-covariance.

    f_H(x) = sum_{i=1}^{m} ρ_i f_{L_i}(x) + δ(x)

    This kernel supports multiple low-fidelity (LF) sources with **learnable**
    cross-covariances between LF sources.
    
    Parameters:
    - kernel_Ls: List of GP kernels for each LF source.
    - kernel_delta: GP kernel for HF discrepancy.
    - num_LF: Number of LF sources.
    - num_output_dims: Number of output dimensions (Y.shape[1]).
    """

    def __init__(self, kernel_Ls, kernel_delta, num_LF, num_output_dims):
        super().__init__()
        self.num_LF = num_LF
        self.kernel_Ls = kernel_Ls  # List of LF kernels
        self.kernel_delta = kernel_delta  # HF discrepancy kernel

        # Trainable scaling factors for LF to HF relationships
        self.rho = gpflow.Parameter(
            np.ones((num_LF, num_output_dims)), transform=positive()
        )  # Shape (num_LF, output_dim)

        # Trainable correlation parameters between LF sources (-1 to 1)
        self.rho_LF = gpflow.Parameter(
            0.5 * np.ones((num_LF, num_LF)), transform=tfp.bijectors.Sigmoid(),
        )  # Shape (num_LF, num_LF), initialized near 0.5

    def K(self, X, X2=None, ith_output_dim=0):
        if X2 is None:
            X2 = X

        X = tf.convert_to_tensor(X, dtype=tf.float64)
        X2 = tf.convert_to_tensor(X2, dtype=tf.float64)

        # Extract fidelity masks
        masks_L = [tf.where(X[:, -1] == i)[:, 0] for i in range(self.num_LF)]
        mask_H = tf.where(X[:, -1] == self.num_LF)[:, 0]
        masks2_L = [tf.where(X2[:, -1] == i)[:, 0] for i in range(self.num_LF)]
        mask2_H = tf.where(X2[:, -1] == self.num_LF)[:, 0]

        # Extract rho values
        rho_i = self.rho[:, ith_output_dim]  # Shape: (num_LF,)

        # Initialize covariance matrix
        K_full = tf.zeros((X.shape[0], X2.shape[0]), dtype=tf.float64)

        # LF-LF covariance
        for i in range(self.num_LF):
            for j in range(self.num_LF):
                X_Li, X_Lj = tf.gather(X[:, :-1], masks_L[i]), tf.gather(X2[:, :-1], masks2_L[j])
                rho_ij = self.rho_LF[i, j] if i != j else 1.0
                K_LF = rho_ij * self.kernel_Ls[i].K(X_Li, X_Lj)

                indices = tf.stack(tf.meshgrid(masks_L[i], masks2_L[j], indexing="ij"), axis=-1)
                K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices, [-1, 2]), tf.reshape(K_LF, [-1]))

        # LF-HF covariance
        if tf.size(mask_H) > 0 and tf.size(mask2_H) > 0:
            for i in range(self.num_LF):
                X_LF, X2_HF = tf.gather(X[:, :-1], masks_L[i]), tf.gather(X2[:, :-1], mask2_H)
                K_LH = self.kernel_Ls[i].K(X_LF, X2_HF) * rho_i[i]
                K_HL = self.kernel_Ls[i].K(X2_HF, X_LF) * rho_i[i]

                indices_LH = tf.stack(tf.meshgrid(masks_L[i], mask2_H, indexing="ij"), axis=-1)
                indices_HL = tf.stack(tf.meshgrid(mask_H, masks2_L[i], indexing="ij"), axis=-1)

                K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_LH, [-1, 2]), tf.reshape(K_LH, [-1]))
                K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HL, [-1, 2]), tf.reshape(K_HL, [-1]))

        # HF-HF covariance
        if tf.size(mask_H) > 0 and tf.size(mask2_H) > 0:
            X_H, X2_H = tf.gather(X[:, :-1], mask_H), tf.gather(X2[:, :-1], mask2_H)
            K_HH = sum([self.kernel_Ls[i].K(X_H, X2_H) * (rho_i[i] ** 2) for i in range(self.num_LF)])
            K_HH += self.kernel_delta.K(X_H, X2_H)

            indices_HH = tf.stack(tf.meshgrid(mask_H, mask2_H, indexing="ij"), axis=-1)
            K_full = tf.tensor_scatter_nd_update(K_full, tf.reshape(indices_HH, [-1, 2]), tf.reshape(K_HH, [-1]))

        # Add jitter for numerical stability
        K_full += tf.eye(tf.shape(K_full)[0], dtype=tf.float64) * 1e-6
    
        return K_full
    

    def K_diag(self, X, ith_output_dim=0):
        """Computes the diagonal elements of the covariance matrix."""
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        masks = [tf.where(X[:, -1] == i)[:, 0] for i in range(self.num_LF + 1)]

        K_diag_full = tf.zeros((X.shape[0],), dtype=tf.float64)
        
        # LF diagonal elements
        for i in range(self.num_LF):
            X_L = tf.gather(X[:, :-1], masks[i])
            K_diag_L = self.kernel_Ls[i].K_diag(X_L)  # Only self-variance
            K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(masks[i], [-1, 1]), tf.reshape(K_diag_L, [-1]))

        # HF diagonal elements
        X_H = tf.gather(X[:, :-1], masks[-1])
        K_diag_H = sum(self.kernel_Ls[i].K_diag(X_H) * (self.rho[i, ith_output_dim] ** 2) for i in range(self.num_LF))
        K_diag_H += self.kernel_delta.K_diag(X_H)
        K_diag_full = tf.tensor_scatter_nd_update(K_diag_full, tf.reshape(masks[-1], [-1, 1]), tf.reshape(K_diag_H, [-1]))

        return K_diag_full

    
class GraphMultiFidelityGPModel(gpflow.models.GPR):
    """
    Gaussian Process model for graph-structured multi-fidelity learning with multiple correlated LF sources.

    This model:
    - Uses the `GraphMultiFidelityKernel` which includes cross-covariance between LF sources.
    - Learns separate `rho[i]` parameters for LF-to-HF relationships.
    - Learns `rho_LF[i, j]` cross-correlation parameters between LF sources.
    - Supports multi-output regression.
    """

    def __init__(self, X, Y, kernel_Ls, kernel_delta):
        num_LF = len(kernel_Ls)  # Number of LF sources
        num_output_dims = Y.shape[1]  # Number of independent outputs
        
        # Initialize the custom multi-fidelity kernel
        self.kernel = GraphMultiFidelityKernel(kernel_Ls, kernel_delta, num_LF, num_output_dims)
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-3)

        super().__init__((X, Y), kernel=self.kernel, likelihood=likelihood)
        set_trainable(self.likelihood.variance, False)

        self.num_LF = num_LF
        self.num_output_dims = num_output_dims

    def optimize(self, max_iters=1000, learning_rate=0.01, use_adam=True, unfix_noise_after=500):
        """
        Optimizes the model while ensuring proper multi-output learning.
        
        - Updates per-output `rho[i]` parameters.
        - Updates cross-covariance parameters `rho_LF[i, j]`.
        - Uses Adam or L-BFGS with noise-fixing for stability.
        """
        self.loss_history = []

        if use_adam:
            optimizer = tf.optimizers.Adam(learning_rate)

            @tf.function
            def optimization_step():
                with tf.GradientTape() as tape:
                    loss = -self.log_marginal_likelihood()  # Maximize log-marginal likelihood
                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return loss

            print("Optimizing with Adam...")
            for i in range(max_iters):
                loss = optimization_step()
                self.loss_history.append(loss.numpy())

                if i == unfix_noise_after:
                    print(f"🔹 Unfixing noise at iteration {i}")
                    set_trainable(self.likelihood.variance, True)

                if i % 100 == 0:
                    print(f"🔹 Iteration {i}: Loss = {-loss.numpy()}")

        else:
            print("Optimizing with L-BFGS (Scipy)...")
            
            def loss_closure():
                loss = -self.log_marginal_likelihood()
                self.loss_history.append(loss.numpy())
                return loss

            scipy_optimizer = gpflow.optimizers.Scipy()
            scipy_optimizer.minimize(loss_closure, self.trainable_variables, options={"maxiter": max_iters})

            set_trainable(self.likelihood.variance, True)
            scipy_optimizer.minimize(loss_closure, self.trainable_variables, options={"maxiter": max_iters})
