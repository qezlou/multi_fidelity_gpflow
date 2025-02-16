import gpflow
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from gpflow.kernels import SquaredExponential, Linear
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.kernels import LinearCoregionalization
from .linear import LinearMultiFidelityKernel  # Your existing LinearMultiFidelityKernel

class SingleBinSVGP(SVGP):
    """
    Multi-Fidelity Sparse Variational GP with:
    - **Inference in Latent GP Space** (L < P)
    - **LinearCoregionalization** for multi-output correlation
    - **Improved Inducing Variable Selection (KMeans)**
    - **Improved Training Stability (Better Initialization & Learning Rate Decay)**
    """

    def __init__(self, X, Y, kernel_L, kernel_delta, num_outputs, Z, random_state=42):
        """
        Initializes the Multi-Fidelity SVGP model.

        Parameters:
            X (np.ndarray): Input data (N, D).
            Y (np.ndarray): Output data (N, P).
            kernel_L (gpflow.kernels.Kernel): Kernel for low-fidelity data.
            kernel_delta (gpflow.kernels.Kernel): Kernel for delta data.
            num_latents (int): Number of latent GPs (L).
            num_outputs (int): Number of output dimensions (P).
            Z (np.ndarray): Inducing point locations (M, D).
            window_fraction (float): Fraction of total outputs each latent covers.
            scale (float): Scaling factor for initial weights.
        """
        self.num_outputs = num_outputs

        # ✅ Multi-Fidelity Kernel
        mf_kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_output_dims=num_outputs)

        # ✅ Use LinearCoregionalization for Multi-Output GP
        kernel_list = [mf_kernel for _ in range(num_outputs)]
        multioutput_kernel = gpflow.kernels.SeparateIndependent(kernel_list)

        # ✅ Use KMeans to Find Good Inducing Points
        kmeans = KMeans(n_clusters=Z.shape[0], random_state=random_state).fit(X)
        Z_init = kmeans.cluster_centers_
        print("🔹 KMeans Inducing Points:", Z_init)
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z_init))

        # ✅ Better Variational Initialization
        q_mu = np.zeros((Z.shape[0], num_outputs))  # M × L
        q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], num_outputs, axis=0) * 0.1  # L × M × M, scaled down

        # ✅ Define SVGP Model
        likelihood = Gaussian()
        super().__init__(kernel=multioutput_kernel, likelihood=likelihood,
                         inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt)

    def optimize(self, data, max_iters=10000, initial_lr=0.005, unfix_noise_after=5000):
        """
        Optimizes the model using Adam with cosine decay.
        """
        X, Y = data
        optimizer = tf.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters))
        self.loss_history = []

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = -self.elbo((X, Y))  # ✅ GPflow’s ELBO computation
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss  # Track loss

        print("🔹 Optimizing...")
        for i in range(max_iters):
            loss = optimization_step()
            self.loss_history.append(loss.numpy())  # Store loss history

            if i == unfix_noise_after:
                print("🔹 Unfixing noise variance at iteration", i)
                gpflow.utilities.set_trainable(self.likelihood.variance, True)
            if i % 1000 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}")
