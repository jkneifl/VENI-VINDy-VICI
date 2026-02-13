"""
Shared utility functions for examples.

This module contains common functions used across different example scripts
(MEMS, reaction_diffusion, etc.) to avoid code duplication.
"""

import os
import random
import logging
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vindy.utils import switch_data_format as switch_data_format


def get_config():
    """
    Import and return the config module.

    This function handles the import of the examples config module with proper
    error handling. Use this in your scripts to avoid repetitive import logic.

    Returns:
        module: The config module object.

    Raises:
        ImportError: If config.py doesn't exist or can't be imported.
    """

    try:
        import examples.config as config

        return config
    except ImportError:
        raise ImportError(
            "Could not import config. "
            "Please ensure that the examples/config.py file exists and is correctly configured. "
            "You can copy examples/config.py.template to examples/config.py and customize it "
            "with the correct data paths and parameters for your setup."
        )


def set_seed(seed: int):
    """
    Set seed for reproducibility in TensorFlow, NumPy, and Python's random module.

    Args:
        seed (int): The seed value to set.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_data_path(data_path: str, zenodo_doi: str = "10.5281/zenodo.18313843"):
    """
    Validate that a data file exists and provide a helpful error message if not.

    Args:
        data_path (str): Path to the data file.
        zenodo_doi (str): Zenodo DOI for downloading the data (default: 10.5281/zenodo.18313843).

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. "
            f"Please download the file from Zenodo (http://doi.org/{zenodo_doi}) and "
            f"specify the correct path in the examples/config.py file."
        )


def plot_train_history(trainhist, result_dir, validation=True):
    """
    Plot training history including loss curves.

    Args:
        trainhist (dict): Training history dictionary containing loss values.
        result_dir (str): Directory to save the plot.
        validation (bool): Whether to include validation loss in the plot.
    """
    try:
        os.makedirs(result_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training loss
        if "loss" in trainhist:
            ax.plot(trainhist["loss"], label="Training Loss", linewidth=2)

        # Plot validation loss if requested and available
        if validation and "val_loss" in trainhist:
            ax.plot(trainhist["val_loss"], label="Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training History", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        plt.show()

        # Save figure
        suffix = "_val" if validation else "_train"
        save_path = os.path.join(result_dir, f"training_history{suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved training history plot to {save_path}")

        plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to plot training history: {e}")


def plot_coefficients_train_history(trainhist, result_dir):
    """
    Plot the evolution of SINDy coefficients during training.

    Args:
        trainhist (dict): Training history dictionary.
        result_dir (str): Directory to save the plot.
    """
    os.makedirs(result_dir, exist_ok=True)

    # Check if coefficient history is available
    if "coeffs_mean" not in trainhist:
        logging.warning("No SINDy coefficients found in training history")
        return

    coeffs = np.array(trainhist["coeffs_mean"])

    # Plot coefficient evolution
    fig, ax = plt.subplots(figsize=(12, 6))

    n_coeffs = coeffs.shape[1] if coeffs.ndim > 1 else 1
    for i in range(n_coeffs):
        if coeffs.ndim > 1:
            ax.plot(coeffs[:, i], label=f"Coeff {i}", linewidth=1.5)
        else:
            ax.plot(coeffs, label=f"Coeff {i}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Coefficient Value", fontsize=12)
    ax.set_title("SINDy Coefficient Evolution", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    save_path = os.path.join(result_dir, "coefficients_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved coefficient history plot to {save_path}")

    plt.close(fig)


def create_result_directory(base_dir: str, model_name: str) -> str:
    """
    Create a result directory for saving model outputs.

    Args:
        base_dir (str): Base directory for results.
        model_name (str): Name of the model/experiment.

    Returns:
        str: Path to the created result directory.
    """
    result_dir = os.path.join(base_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f"Result directory: {result_dir}")
    return result_dir


def log_model_summary(veni, result_dir: str = None):
    """
    Log a summary of the VENI model architecture.

    Args:
        veni: The VENI model instance.
        result_dir (str, optional): Directory to save the summary text file.
    """
    try:
        logging.info("=" * 60)
        logging.info("Model Summary:")
        logging.info("=" * 60)

        # Log key parameters
        logging.info(f"Reduced order: {veni.reduced_order}")
        logging.info(f"Second order: {veni.second_order}")
        logging.info(f"Scaling method: {veni.scaling}")

        # Get model summary
        summary_lines = []
        veni.summary(print_fn=lambda x: summary_lines.append(x))

        for line in summary_lines:
            logging.info(line)

        # Save to file if directory provided
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            summary_path = os.path.join(result_dir, "model_summary.txt")
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_lines))
            logging.info(f"Model summary saved to {summary_path}")

        logging.info("=" * 60)
    except Exception as e:
        logging.warning(f"Failed to log model summary: {e}")


def get_latent_initial_conditions(veni, x, dxdt, dxddt, mean_or_sample):
    """
    Compute the initial conditions in the latent space for integration based on the provided state and its derivatives.

    Args:
    veni: The VENI model instance.
    X: The state data array.
    DXDT: The first time derivative of the state data array.
    DXDDT: The second time derivative of the state data array (can be None if
        not available).
    mean_or_sample: Whether to compute the mean initial condition or sample from the distribution ("mean" or "sample").
    """
    if veni.second_order:
        z0, dz0, _ = veni.calc_latent_time_derivatives(
            x, dxdt, dxddt, mean_or_sample=mean_or_sample
        )
        ic = np.concatenate(
            [z0[0], dz0[0]], axis=-1
        )  # remove batch dimension and concatenate
    else:
        z0, dz0 = veni.calc_latent_time_derivatives(
            x, dxdt, None, mean_or_sample=mean_or_sample
        )
        ic = z0[0]  # remove batch dimension

    return ic


def perform_inference(
    veni,
    sim_ids,
    n_sims,
    n_timesteps,
    t,
    x,
    dxdt=None,
    params=None,
):
    """
    Perform inference on test trajectories and plot the results.

    Args:
        veni: The trained VENI model.
        x: Scaled test data.
        dxdt: Scaled test data derivatives.
        t: Test time steps.
        params: Test parameters.
        sim_ids: List of test trajectory indices.
        n_sims: Number of simulations.
        n_timesteps: Number of timesteps in each test trajectory.

    Returns:
        Tuple: Predicted trajectories and their corresponding time steps.
    """
    # Reshape data into simulation-wise format
    T = switch_data_format(t, n_sims, n_timesteps, target_format="3d")
    z, dzdt = veni.calc_latent_time_derivatives(x, dxdt)
    Z = switch_data_format(z, n_sims, n_timesteps, target_format="3d")
    DZDT = switch_data_format(dzdt, n_sims, n_timesteps, target_format="3d")
    Params = switch_data_format(params, n_sims, n_timesteps, target_format="3d")

    z_preds = []
    t_preds = []
    start_time = datetime.datetime.now()
    for i, j in enumerate(sim_ids):
        logging.info(f"Processing trajectory {i+1}/{len(sim_ids)}")
        # Perform integration
        ic = np.concatenate([Z[j, 0], DZDT[j, 0]]) if veni.second_order else Z[j, 0]
        sol = veni.integrate(
            ic,
            T[j].squeeze(),
            mu=Params[j] if params is not None else None,
        )
        z_preds.append(sol.y)
        t_preds.append(sol.t)
    end_time = datetime.datetime.now()
    logging.info(
        f"Inference time: {(end_time - start_time).total_seconds()/len(sim_ids):.2f} seconds per trajectory"
    )

    # Convert predictions to arrays
    z_preds = np.array(z_preds)
    t_preds = np.array(t_preds)

    return Z, z_preds, t_preds


def plot_inference_results(t_preds, z_preds, T, Z, sim_ids, state_id=0):

    # Plot inference results
    fig, axs = plt.subplots(len(sim_ids), 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Inference of Test Trajectories")
    for i, j in enumerate(sim_ids):
        axs[i].set_title(f"Test Trajectory {j}")
        axs[i].plot(T[j], Z[j][:, state_id], color="blue", label="True")
        axs[i].plot(
            t_preds[i],
            z_preds[i][state_id],
            color="red",
            linestyle="--",
            label="Predicted",
        )
        axs[i].set_xlabel("$t$")
        axs[i].set_ylabel("$z$")
        axs[i].legend()
    plt.tight_layout()
    plt.show()


def perform_forward_uq(
    veni,
    sim_ids,
    n_traj,
    n_sims,
    n_timesteps,
    t,
    x,
    dxdt,
    dxddt=None,
    params=None,
    sigma=3,
):
    """
    Perform forward uncertainty quantification by sampling trajectories from the SINDy model.
    The function is flexible with optional `dxddt` and `params` (pass None if not available).

    Args:
        veni: The VENI model instance.
        sim_ids: List of simulation indices to process.
        n_traj: Number of trajectories to sample for each simulation.
        n_sims: Total number of simulations in the dataset.
        n_timesteps: Number of timesteps in each simulation.
        t: Time vector
        x: State data
        dxdt: First time derivative of state data.
        dxddt: Second time derivative of state data (optional).
        params: Additional parameters for integration (optional).
        sigma: Number of standard deviations for confidence intervals.


    Returns a dictionary with keys identical to the MEMS implementation:
    sampled_times, sampled_latent_trajectories, mean_latent_samples,
    std_latent_samples, mean_latent, lower_bound_latent, upper_bound_latent,
    z, dzdt_test
    """

    second_order = veni.second_order

    # Accept tensors or numpy arrays
    def _to_numpy(a):
        if a is None:
            return None
        try:
            return a.numpy()
        except Exception:
            return np.asarray(a)

    x = _to_numpy(x)
    dxdt = _to_numpy(dxdt)
    dxddt = _to_numpy(dxddt)
    params = _to_numpy(params)
    t = _to_numpy(t)

    # Switch to simulation-wise 3D arrays if needed
    X = switch_data_format(x, n_sims, n_timesteps, target_format="3d")
    DXDT = switch_data_format(dxdt, n_sims, n_timesteps, target_format="3d")
    DXDDT = (
        switch_data_format(dxddt, n_sims, n_timesteps, target_format="3d")
        if second_order
        else None
    )
    T = switch_data_format(t, n_sims, n_timesteps, target_format="3d")
    Params = (
        switch_data_format(params, n_sims, n_timesteps, target_format="3d")
        if params is not None
        else None
    )

    # compute latent derivatives from the provided (possibly vectorized) arrays
    if second_order:
        z, dzdt, _ = veni.calc_latent_time_derivatives(x, dxdt, dxddt)
    else:
        z, dzdt = veni.calc_latent_time_derivatives(x, dxdt, None)

    # ensure z and dzdt have simulation-wise shapes (n_sims, n_timesteps, n_states)
    Z = switch_data_format(_to_numpy(z), n_sims, n_timesteps, target_format="3d")
    DZDT = switch_data_format(_to_numpy(dzdt), n_sims, n_timesteps, target_format="3d")

    # Save kernel state
    kernel_orig, kernel_scale_orig = (
        veni.sindy_layer.kernel,
        veni.sindy_layer.kernel_scale,
    )

    sampled_times = []
    latent_trajectories_samples = []
    mean_latent_trajectories = []

    for i in sim_ids:
        logging.info("Processing trajectory %d/%d", i + 1, len(sim_ids))

        # mu parameter for SINDy integration
        mu = Params[i] if params is not None else None

        # time vector for integration
        tvec = T[i].squeeze()

        # Get initial conditions in physical space for the first timestep of each simulation
        x0, dx0dt0, dx0ddt0 = (
            X[i, 0:1],
            DXDT[i, 0:1],
            DXDDT[i, 0:1] if second_order else None,
        )

        # sampling
        traj_samples = []
        traj_times = []
        for traj in range(n_traj):
            logging.info("\tSampling model %d/%d", traj + 1, n_traj)

            # sample initial condition
            ic = get_latent_initial_conditions(
                veni, x0, dx0dt0, dx0ddt0, mean_or_sample="sample"
            )

            sol, coeffs = veni.sindy_layer.integrate_uq(ic, tvec, mu=mu)

            traj_samples.append(np.asarray(sol.y))
            traj_times.append(np.asarray(sol.t))

        sampled_times.append(traj_times)
        latent_trajectories_samples.append(traj_samples)

        # Mean / nominal integration (using original kernel)
        veni.sindy_layer.kernel, veni.sindy_layer.kernel_scale = (
            kernel_orig,
            kernel_scale_orig,
        )
        # mean initial condition (using mean prediction from encoder)
        ic = get_latent_initial_conditions(
            veni, x0, dx0dt0, dx0ddt0, mean_or_sample="sample"
        )

        sol_mean = veni.integrate(ic, tvec, mu=mu)
        mean_latent_trajectories.append(np.asarray(sol_mean.y))

    # convert lists to arrays
    latent_trajectories_samples = np.array(latent_trajectories_samples)

    # expected shape: (ns, n_traj, n_states, n_timesteps) -> transpose to (ns, n_traj, n_timesteps, n_states)
    latent_trajectories_samples = np.transpose(
        latent_trajectories_samples, (0, 1, 3, 2)
    )

    # statistics across sampled trajectories: mean/std over axis=1 (samples)
    mean_latent_samples = np.mean(latent_trajectories_samples, axis=1)
    std_latent_samples = np.std(latent_trajectories_samples, axis=1)

    # mean trajectories: convert list to array and ensure shape (ns, n_timesteps, n_states)
    mean_latent = np.array(mean_latent_trajectories)
    mean_latent = np.transpose(mean_latent, (0, 2, 1))

    lower_bound_latent = mean_latent - sigma * std_latent_samples
    upper_bound_latent = mean_latent + sigma * std_latent_samples

    return {
        "sampled_times": sampled_times,
        "latent_trajectories_samples": latent_trajectories_samples,
        "mean_latent_samples": mean_latent_samples,
        "std_latent_samples": std_latent_samples,
        "mean_latent": mean_latent,
        "lower_bound_latent": lower_bound_latent,
        "upper_bound_latent": upper_bound_latent,
        "z": Z,
        "dzdt": DZDT,
    }


def uq_plots(
    sampled_times,
    mean_latent,
    mean_latent_samples,
    std_latent_samples,
    t_test,
    z_test,
    test_ids,
    state_id=0,
):
    """
    Generate UQ plots.

    Args:
        sampled_times (list): Time points for sampled UQ trajectories.
        mean_latent (list): Mean trajectories from deterministic integration.
        mean_latent_samples (list): Mean of sampled trajectories.
        std_latent_samples (list): Standard deviation of sampled trajectories.
        t_test (np.ndarray): Test time steps.
        z_test (np.ndarray): Latent states for test data.
        test_ids (list): List of test trajectory indices to plot.
    """
    n_test = len(test_ids)
    # plot the mean and 3*std of the trajectories
    fig, axs = plt.subplots(n_test, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Integrated Test Trajectories")
    for i, i_test in enumerate(test_ids):
        axs[i].set_title(f"Test Trajectory {i_test + 1}")
        # for i in range(2):
        axs[i].plot(t_test[i_test], z_test[i_test][:, state_id], color="blue")
        axs[i].plot(
            sampled_times[i][0],
            mean_latent[i, :, state_id],
            color="red",
            linestyle="--",
        )
        axs[i].fill_between(
            sampled_times[i][0],
            mean_latent_samples[i][:, state_id]
            - 3 * std_latent_samples[i][:, state_id],
            mean_latent_samples[i][:, state_id]
            + 3 * std_latent_samples[i][:, state_id],
            color="red",
            alpha=0.3,
        )
        axs[i].set_xlabel("$t$")
        axs[i].set_ylabel("$z$")

    plt.tight_layout()
    plt.show()
