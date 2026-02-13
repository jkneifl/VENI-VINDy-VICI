"""
BEAM MODEL SCRIPT

This script trains and evaluates a MEMS model using the VENI framework.
The model identifies the dynamics of a MEMS system based on input data.
The data is assumed to be preprocessed and available in the specified path (as per config.py).

Model:
    dzdt    = z
    dzddt   = -w0^2 * z - 2 * xi * w0 * z_dot - gamma * z^3 + u
            = - 0.29975625 * z - 0.01095 z_dot - gamma * z^3 + u
"""

import os
import logging
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from vindy import VENI
from vindy.libraries import PolynomialLibrary, ForceLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Laplace
from vindy.callbacks import SaveCoefficientsCallback
from vindy.utils import switch_data_format, coefficient_distribution_gif
from examples.MEMS.utils import load_mems_data

# Import shared utilities
from vindy.utils import (
    set_seed,
    plot_train_history,
    plot_coefficients_train_history,
    get_config,
    perform_inference,
    plot_inference_results,
    perform_forward_uq,
    uq_plots,
)

# Import configuration (data paths)
config = get_config()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Constants
LOAD_MODEL = True
CREATE_GIF = False
BETA_VINDY = 1e-8  # VINDy prior weight
BETA_VAE = 1e-8  # VAE KL loss weight
L_REC = 1e-3  # reconstruction loss weight
L_DZ = 1e0  # latent derivative loss weight
L_DX = 1e-5  # physical derivative loss weight
END_TIME_STEP = 14000  # until which time step the data is used for training
MODEL_NAME = "MEMS"
IDENTIFICATION_LAYER = "vindy"  # 'vindy' or 'sindy'
REDUCED_ORDER = 1  # latent space dimension
PCA_ORDER = 3  # PCA order for data preprocessing
NTH_TIME_STEP = 3  # use every nth time step for training
EPOCHS = 500  # number of training epochs
BATCH_SIZE = 256  # training batch size
LEARNING_RATE = 2e-3  # learning rate
SECOND_ORDER = True  # use second order dynamics
PDF_THRESHOLD = 5  # PDF threshold for coefficient sparsification
SEED = 42  # random seed for reproducibility


def visualize_sample_data(t, x, dxdt, dxddt, params, n_timesteps):
    """
    Visualize a sample of the training data.

    Args:
        t (np.ndarray): Time steps.
        x (np.ndarray): State data.
        dxdt (np.ndarray): State derivatives.
        dxddt (np.ndarray): State second derivatives.
        params (np.ndarray): Parameters.
        n_timesteps (int): Number of time steps per simulation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("MEMS Beam Training Data Sample", fontsize=14, fontweight="bold")

    # Select first simulation for visualization
    sim_length = n_timesteps
    sample_time = t[:sim_length]
    sample_x = x[:sim_length, 0]  # First PCA component
    sample_dxdt = dxdt[:sim_length, 0]
    sample_dxddt = dxddt[:sim_length, 0]
    sample_params = params[:sim_length] if params.shape[1] > 0 else None

    # Position vs time
    axes[0, 0].plot(sample_time, sample_x, "b-", linewidth=1.5)
    axes[0, 0].set_title("Beam Position of 1st PCA mode")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity vs time
    axes[0, 1].plot(sample_time, sample_dxdt, "r-", linewidth=1.5)
    axes[0, 1].set_title("Beam Velocity of 1st PCA mode")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].grid(True, alpha=0.3)

    # Acceleration vs time
    axes[0, 2].plot(sample_time, sample_dxddt, "m-", linewidth=1.5)
    axes[0, 2].set_title("Beam Acceleration of 1st PCA mode")
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel("Acceleration")
    axes[0, 2].grid(True, alpha=0.3)

    # Phase portrait
    axes[1, 0].plot(sample_x, sample_dxdt, "g-", linewidth=1, alpha=0.7)
    axes[1, 0].scatter(
        sample_x[0], sample_dxdt[0], color="green", s=50, label="Start", zorder=5
    )
    axes[1, 0].scatter(
        sample_x[-1], sample_dxdt[-1], color="red", s=50, label="End", zorder=5
    )
    axes[1, 0].set_title("Phase Portrait")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Velocity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # forcing function is u(t) = F*cos(omega*t)
    forcing = ForceLibrary(functions=[tf.cos])(sample_params)
    # Parameter forcing (if available)
    if sample_params is not None:
        axes[1, 1].plot(sample_time, forcing, "m-", linewidth=1.5)
        axes[1, 1].set_title("External Forcing")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Force Parameter")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No parameter data\navailable",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("External Forcing")

    plt.tight_layout()
    plt.show()


def create_model(x, params, dt, n_dof):
    """
    Create the VENI model.

    Args:
        params (np.ndarray): Parameters for the model.
        dt (float): Time step size.
        n_dof (int): Number of degrees of freedom.

    Returns:
        VENI: The initialized VENI model.
    """
    logging.info("Creating model...")
    libraries = [PolynomialLibrary(3)]
    param_libraries = [ForceLibrary(functions=[tf.cos])]

    layer_params = dict(
        state_dim=REDUCED_ORDER,
        param_dim=params.shape[1],
        feature_libraries=libraries,
        second_order=SECOND_ORDER,
        param_feature_libraries=param_libraries,
        x_mu_interaction=False,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-8, l2=0),
        mask=None,
        fixed_coeffs=None,
    )

    if IDENTIFICATION_LAYER == "vindy":
        sindy_layer = VindyLayer(
            beta=BETA_VINDY,
            priors=Laplace(0.0, 1.0),
            **layer_params,
        )
    elif IDENTIFICATION_LAYER == "sindy":
        sindy_layer = SindyLayer(**layer_params)
    else:
        raise ValueError('IDENTIFICATION_LAYER must be either "vindy" or "sindy"')

    return VENI(
        sindy_layer=sindy_layer,
        beta=BETA_VAE * REDUCED_ORDER / n_dof,
        reduced_order=REDUCED_ORDER,
        x=x,
        mu=params,
        scaling="individual_sqrt",
        second_order=SECOND_ORDER,
        layer_sizes=[32, 32, 32],
        activation="elu",
        l_rec=L_REC,
        l_dz=L_DZ,
        l_dx=L_DX,
        dt=dt,
    )


def train_model(veni, x_input, x_input_val, weights_path, log_dir, train_histdir):
    """
    Train the VENI model.

    Args:
        veni (VENI): The VENI model.
        x_input (list): Training data.
        x_input_val (list): Validation data.
        weights_path (str): Path to save the model weights.
        log_dir (str): Directory for TensorBoard logs.
    """

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if LOAD_MODEL:
        logging.info("Loading model...")
        veni.load_weights(os.path.join(weights_path))
    else:
        logging.info("Training model...")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                verbose=0,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            SaveCoefficientsCallback(),
        ]

        import time

        start_time = time.time()
        trainhist = veni.fit(
            x=x_input,
            validation_data=(x_input_val, None),
            callbacks=callbacks,
            y=None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        end_time = time.time()
        logging.info(f"time per epoch: {(end_time - start_time)/EPOCHS:.2f} seconds")
        # Save training history
        np.save(
            train_histdir,
            trainhist.history,
        )

        veni.print(precision=4)

        # save model
        veni.load_weights(weights_path)

    # load trainhist
    trainhist = np.load(
        train_histdir,
        allow_pickle=True,
    ).item()

    return trainhist


def training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni):
    """
    Generate training plots and visualizations.
    Args:
        trainhist (dict): Training history.
        result_dir (str): Directory to save results.
        x_train_scaled (np.ndarray): Scaled training data.
        x_test_scaled (np.ndarray): Scaled test data.
        veni (VENI): The trained VENI model.
    """
    # Plot training history
    plot_train_history(trainhist, result_dir, validation=True)
    plot_coefficients_train_history(trainhist, result_dir)

    # reconstruction of PCA trajectories
    veni.vis_modes(x_test_scaled, 4)
    veni.vis_modes(x_train_scaled, 4)

    # visualize identified coefficients
    veni.sindy_layer.visualize_coefficients(x_range=[-1.5, 1.5])
    plt.show()


def main():
    """
    Main function to load data, create the model, train, and evaluate.
    """

    # Set seed for reproducibility
    set_seed(SEED)

    # Load data
    (
        t,
        params,
        x,
        dxdt,
        dxddt,
        t_test,
        params_test,
        x_test,
        dxdt_test,
        dxddt_test,
        ref_coords,
        V,
        n_sims,
        n_timesteps,
    ) = load_mems_data(
        config.mems,
        end_time_step=END_TIME_STEP,
        nth_time_step=NTH_TIME_STEP,
        pca_order=PCA_ORDER,
    )

    n_timesteps_test = x_test.shape[0] // n_sims
    n_dof = x.shape[1]
    dt = t[1] - t[0]

    # Visualize sample training data
    visualize_sample_data(t, x, dxdt, dxddt, params, n_timesteps)

    # Create model
    veni = create_model(x, params, dt, n_dof)

    # Scale data
    veni.define_scaling(x)
    x_train_scaled, dxdt_train_scaled, dxddt_train_scaled = (
        veni.scale(x).numpy(),
        veni.scale(dxdt).numpy(),
        veni.scale(dxddt).numpy(),
    )
    x_test_scaled, dxdt_test_scaled, dxddt_test_scaled = (
        veni.scale(x_test).numpy(),
        veni.scale(dxdt_test).numpy(),
        veni.scale(dxddt_test).numpy(),
    )

    x_input = [
        x_train_scaled[: 24 * n_timesteps],
        dxdt_train_scaled[: 24 * n_timesteps],
        dxddt_train_scaled[: 24 * n_timesteps],
        params[: 24 * n_timesteps],
    ]
    x_input_val = [
        x_train_scaled[24 * n_timesteps :],
        dxdt_train_scaled[24 * n_timesteps :],
        dxddt_train_scaled[24 * n_timesteps :],
        params[24 * n_timesteps :],
    ]

    # Compile and build model
    veni.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    veni.build(input_shape=([input.shape for input in x_input], None))

    # Train model
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    log_dir = os.path.join(
        result_dir,
        f'{MODEL_NAME}/log/{MODEL_NAME}_{REDUCED_ORDER}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}',
    )
    weights_path = os.path.join(
        result_dir,
        f"{MODEL_NAME}/{MODEL_NAME}_{REDUCED_ORDER}_{veni.__class__.__name__}_{IDENTIFICATION_LAYER}.weights.h5",
    )
    train_hist_dir = os.path.join(
        result_dir, f"{MODEL_NAME}/trainhist_{IDENTIFICATION_LAYER}.npy"
    )
    trainhist = train_model(
        veni, x_input, x_input_val, weights_path, log_dir, train_hist_dir
    )

    training_plots(trainhist, result_dir, x_train_scaled, x_test_scaled, veni)

    # coefficients gif
    if CREATE_GIF:
        coefficient_distribution_gif(
            trainhist["coeffs_mean"],
            trainhist["coeffs_scale"],
            veni.sindy_layer,
            os.path.join(result_dir, f"{MODEL_NAME}/coefficients")
        )

    # Sparsification of the identified model
    veni.sindy_layer.pdf_thresholding(threshold=PDF_THRESHOLD)

    # Inference and forward UQ
    logging.info("Performing inference and forward UQ...")

    # Predict latent states and uncertainty bounds
    n_traj = 10
    test_ids = [1, 10]

    # VICI
    # Inference
    Z, z_preds, t_preds = perform_inference(
        veni,
        test_ids,
        n_sims,
        n_timesteps_test,
        t_test,
        x_test_scaled,
        dxdt_test_scaled,
        params_test,
    )
    T = switch_data_format(t_test, n_sims, n_timesteps_test, target_format="3d")
    plot_inference_results(t_preds, z_preds, T, Z, test_ids)

    # UQ
    uq_results = perform_forward_uq(
        veni,
        test_ids,
        n_traj,
        n_sims,
        n_timesteps_test,
        t_test,
        x_test_scaled,
        dxdt_test_scaled,
        dxddt_test_scaled,
        params_test,
    )

    # Plot results
    uq_plots(
        uq_results["sampled_times"],
        uq_results["mean_latent"],
        uq_results["mean_latent_samples"],
        uq_results["std_latent_samples"],
        switch_data_format(t_test, n_sims, n_timesteps_test, target_format="3d"),
        uq_results["z"],
        test_ids,
    )


if __name__ == "__main__":
    main()
