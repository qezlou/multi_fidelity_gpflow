import gpflow
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from gpflow.kernels import SquaredExponential, Linear
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from .linear import LinearMultiFidelityKernel  # Your existing LinearMultiFidelityKernel
import pickle
from copy import deepcopy

class SingleBinSVGP(SVGP):
    """
    Multi-Fidelity Sparse Variational GP with:
    - **Inducing Variable Selection with KMeans** for robust initialization.
    - **Stable Optimization** using better parameter initialization.
    """

    def __init__(self, X, Y, kernel_L, kernel_delta, num_outputs, Z, random_state=42):
        """
        Initializes the Multi-Fidelity SVGP model.

        Parameters:
            X (np.ndarray): Input data of shape `(N, D)`, where `D` is the input dimension.
            Y (np.ndarray): Output data of shape `(N, P)`, where `P` is the number of output dimensions.
            kernel_L (gpflow.kernels.Kernel): Kernel for low-fidelity (LF) data.
            kernel_delta (gpflow.kernels.Kernel): Kernel for the high-fidelity (HF) discrepancy.
            num_outputs (int): Number of output dimensions `P` (e.g., 49 bins).
            Z (np.ndarray): Initial inducing points of shape `(M, D)`, where `M` is the number of inducing points.
            random_state (int): Random seed for reproducibility.
        """
        self.num_outputs = num_outputs

        # ✅ Multi-Fidelity Kernel
        # mf_kernel = LinearMultiFidelityKernel(kernel_L, kernel_delta, num_output_dims=num_outputs)
        # ✅ Multi-Fidelity Kernel (Create Independent Instances)

        kernel_list = [LinearMultiFidelityKernel(deepcopy(kernel_L), deepcopy(kernel_delta), num_output_dims=1) for _ in range(num_outputs)]
        # kernel_list = [
        #     LinearMultiFidelityKernel(kernel_L, kernel_delta, num_output_dims=1)
        #     for _ in range(num_outputs)
        # ]

        # ✅ Use LinearCoregionalization for Multi-Output GP (L < P)
        # kernel_list = [mf_kernel for _ in range(num_outputs)]
        multioutput_kernel = gpflow.kernels.SeparateIndependent(kernel_list)

        # ✅ Use KMeans to Find Good Inducing Points
        kmeans = KMeans(n_clusters=Z.shape[0], random_state=random_state).fit(X)
        Z_init = kmeans.cluster_centers_
        print("🔹 KMeans Inducing Points:", Z_init)
        inducing_variable = SharedIndependentInducingVariables(InducingPoints(Z_init))

        # ✅ Variational Parameters Initialization
        q_mu = np.zeros((Z.shape[0], num_outputs))  # M × L
        q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], num_outputs, axis=0) * 0.1  # L × M × M, scaled down

        # ✅ Define SVGP Model
        likelihood = Gaussian()
        super().__init__(kernel=multioutput_kernel, likelihood=likelihood,
                         inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt)

    def optimize(self, data, max_iters=10000, initial_lr=0.01, unfix_noise_after=5000):
        """
        Optimizes the model using Adam with cosine decay.

        Parameters:
            data (tuple): Tuple `(X, Y)`, where:
                - `X` is the training input `(N, D)`.
                - `Y` is the training output `(N, P)`.
            max_iters (int): Maximum number of optimization iterations.
            initial_lr (float): Initial learning rate.
            unfix_noise_after (int): Iteration at which to allow noise variance to be learned.
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
            if i % 10 == 0:
                print(f"🔹 Iteration {i}: ELBO = {-self.elbo((X, Y)).numpy()}")

    def save_model(self, filename="svgp_model.pkl"):
        """
        Saves the trained SVGP model using GPflow's `utilities` and Pickle.

        Parameters:
            filename (str): Path to save the model.
        """
        params = gpflow.utilities.parameter_dict(self)
        with open(filename, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ Model saved to {filename}")

    @staticmethod
    def load_model(filename, X, Y, kernel_L, kernel_delta, num_outputs, Z):
        """
        Loads an SVGP model from a saved file.

        Parameters:
            filename (str): Path to the saved model.
            X (np.ndarray): Training input data (for re-initialization).
            Y (np.ndarray): Training output data (for re-initialization).
            kernel_L (gpflow.kernels.Kernel): Kernel for low-fidelity (LF) data.
            kernel_delta (gpflow.kernels.Kernel): Kernel for HF discrepancy.
            num_outputs (int): Number of output dimensions.
            num_latents (int): Number of latent GPs.
            Z (np.ndarray): Inducing points.

        Returns:
            SingleBinSVGP: The loaded model instance.
        """
        with open(filename, "rb") as f:
            params = pickle.load(f)

        model = SingleBinSVGP(X, Y, kernel_L, kernel_delta, num_outputs, Z)
        gpflow.utilities.multiple_assign(model, params)
        print(f"✅ Model loaded from {filename}")
        return model
