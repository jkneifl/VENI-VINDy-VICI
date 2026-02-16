import numpy as np
import os
import matplotlib.pyplot as plt
import imageio


def add_lognormal_noise(trajectory, sigma):
    noise = np.random.lognormal(mean=0, sigma=sigma, size=trajectory.shape)
    return trajectory * noise, noise


def coefficient_distributions_to_csv(sindy_layer, outdir, var_names=[], param_names=[]):
    """
    Save the coefficient distributions of the SINDy layer to csv files
    :param sindy_layer:
    :param outdir:
    :param var_names:
    :return:
    """
    if not var_names:
        var_names = [f"z{i}" for i in range(1, sindy_layer.output_dim + 1)]
    if not param_names:
        param_names = [f"p_{i}" for i in range(1, sindy_layer.param_dim + 1)]
    feature_names = [
        name_.replace("*", "")
        for name_ in sindy_layer.get_feature_names(var_names, param_names)
    ]
    n_vars = sindy_layer.state_dim
    n_features = len(feature_names)
    _, mean, log_scale = sindy_layer._coeffs
    # reverse log_scale
    scale = sindy_layer.priors.reverse_log(log_scale.numpy())

    mean_values, scale_values = (
        mean.numpy().reshape(n_vars, n_features).T,
        scale.numpy().reshape(n_vars, n_features).T,
    )

    # minimum scale value for which Laplacian dist can be plotted in pgfplots is 1e-4
    scale_values = np.maximum(scale_values, 1e-4)

    for i in range(n_vars):
        save_value = np.concatenate(
            [
                np.array(feature_names)[:, np.newaxis],
                mean_values[:, i : i + 1],
                scale_values[:, i : i + 1],
            ],
            axis=1,
        ).T
        np.savetxt(
            os.path.join(outdir, f"vindy_{var_names[i]}_dot.csv"),
            save_value,
            delimiter=",",
            fmt="%s",
            comments="",
            header=",".join(np.array(range(len(feature_names))).astype(str)),
        )


def coefficient_distribution_gif(
    mean_over_epochs, scale_over_epochs, sindy_layer, outdir
):
    """
    Create a gif showing how the coefficient distributions evolve over time
    """
    os.makedirs(os.path.join(outdir, "coefficients"), exist_ok=True)
    # determine how many frames to generate (capped at 401 frames: indices 0-400)
    max_frames = min(len(mean_over_epochs), len(scale_over_epochs), 401)
    # create gif showing how the coefficient distributions evolve over time
    for i, (mean_, scale_) in enumerate(zip(mean_over_epochs, scale_over_epochs)):
        if i >= max_frames:
            break
        x_range = 1.5  # - (1.5 * i / len(mean_over_epochs))
        # dont show figure
        fig = sindy_layer._visualize_coefficients(
            mean_, scale_, x_range=[-x_range, x_range], y_range=[0, 6]
        )
        # fig title
        fig.suptitle(f"Epoch {i}")
        # save fig as frame for gif
        fig.savefig(os.path.join(outdir, "coefficients", f"coeffs_{i}.png"))
        plt.close(fig)
    # make gif from frames
    images = []
    for i in range(max_frames):
        images.append(
            imageio.imread(os.path.join(outdir, "coefficients", f"coeffs_{i}.png"))
        )
    imageio.mimsave(
        os.path.join(outdir, "coefficients", "coeffs.gif"),
        images,
        duration=100,
    )


def plot_train_history(history, outdir, validation: bool = True):
    """
    Plot the training history
    :param history:
    :param outdir:
    :return:
    """
    os.makedirs(outdir, exist_ok=True)
    # plot training history
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for loss_term, loss_values in history.items():
        if (
            (not validation and "val_" in loss_term)
            or (validation and "val_" not in loss_term)
        ) and "coeffs" not in loss_term:
            ax.plot(loss_values, label=loss_term)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()
    fig.savefig(os.path.join(outdir, "training_history.png"))
    plt.close(fig)


def plot_coefficients_train_history(history, outdir):
    """
    Plot the coefficient training history
    :param history:
    :param outdir:
    :return:
    """
    mean_over_epochs = np.array(history["coeffs_mean"]).squeeze()
    scale_over_epochs = np.array(history["coeffs_scale"]).squeeze()
    os.makedirs(outdir, exist_ok=True)
    # plot training history
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(mean_over_epochs)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Coefficient mean")
    ax[1].plot(scale_over_epochs)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Coefficient scale")
    plt.show()
    fig.savefig(os.path.join(outdir, "coefficients_history.png"))
    plt.close(fig)


def switch_data_format(
    data, n_sims, n_timesteps, spatial_shape=None, target_format="auto"
):
    """
    Convert between vectorized (2D), simulation-wise flattened (3D), and full spatial (5D) data formats.

    Parameters
    - data: np.ndarray. One of:
        * 2D: (n_sims * n_timesteps, features)
        * 3D: (n_sims, n_timesteps, features)
        * 5D: (n_sims, n_timesteps, Nx, Ny, channels)
    - n_sims: int, number of simulations
    - n_timesteps: int, timesteps per simulation
    - spatial_shape: optional tuple describing spatial dims. Accepts (Nx, Ny, channels) or (N, channels) or (Nx, Ny).
      When converting to/from 5D, this is required unless it can be inferred unambiguously from feature size.
    - target_format: 'auto' (default), '2d', '3d', or '5d'. When 'auto', the function chooses a sensible target based on input.

    Returns
    - Converted np.ndarray in requested format.

    Examples
    - 2D -> 5D: provide spatial_shape=(Nx,Ny,channels) and target_format='5d'
    - 5D -> 2D: target_format='2d' or rely on 'auto' to get 3D flattened by default
    """
    if data is None:
        return None

    if target_format not in ("auto", "2d", "3d", "5d"):
        raise ValueError("target_format must be one of 'auto','2d','3d','5d'")

    # Input is vectorized 2D: (n_sims * n_timesteps, features)
    if data.ndim == 2 and data.shape[0] == n_sims * n_timesteps:
        if target_format == "2d" or (target_format == "auto" and data.ndim == 2):
            return data
        features = data.shape[1]
        if target_format == "5d":
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape (Nx,Ny,channels) is required to reshape to 5D"
                )
            # accept (Nx,Ny,channels) or (N,channels)
            if len(spatial_shape) == 3:
                Nx, Ny, channels = spatial_shape
            elif len(spatial_shape) == 2:
                # (N, channels)
                N, channels = spatial_shape
                # try to factor N into Nx,Ny by assuming square grid
                Nx = int(np.sqrt(N))
                if Nx * Nx != N:
                    raise ValueError(
                        "Cannot infer Nx,Ny from N; provide (Nx,Ny,channels)"
                    )
                Ny = Nx
            else:
                raise ValueError("spatial_shape must be length 2 or 3")
            if features != Nx * Ny * channels:
                raise ValueError(
                    f"Feature size {features} does not match provided spatial_shape {spatial_shape}"
                )
            return data.reshape(n_sims, n_timesteps, Nx, Ny, channels)
        # default: to 3D flattened features
        return data.reshape(n_sims, n_timesteps, -1)

    # Input is simulation-wise flattened 3D: (n_sims, n_timesteps, features)
    if data.ndim == 3 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        if target_format == "3d" or (
            target_format == "auto" and data.ndim == 3 and spatial_shape is None
        ):
            return data
        if target_format == "2d":
            return data.reshape(-1, data.shape[-1])
        # convert to 5D
        features = data.shape[2]
        if spatial_shape is None:
            # try infer square grid and single channel
            Nx = int(np.sqrt(features))
            if Nx * Nx == features:
                Ny = Nx
                channels = 1
            else:
                raise ValueError("spatial_shape required to reshape 3D to 5D")
        else:
            if len(spatial_shape) == 3:
                Nx, Ny, channels = spatial_shape
            elif len(spatial_shape) == 2:
                N, channels = spatial_shape
                Nx = int(np.sqrt(N))
                if Nx * Nx != N:
                    raise ValueError(
                        "Cannot infer Nx,Ny from N; provide (Nx,Ny,channels)"
                    )
                Ny = Nx
            else:
                raise ValueError("spatial_shape must be length 2 or 3")
            if features != Nx * Ny * channels:
                raise ValueError(
                    f"Feature size {features} does not match provided spatial_shape {spatial_shape}"
                )
        return data.reshape(n_sims, n_timesteps, Nx, Ny, channels)

    # Input is full spatial 5D: (n_sims, n_timesteps, Nx, Ny, channels)
    if data.ndim == 5 and data.shape[0] == n_sims and data.shape[1] == n_timesteps:
        if target_format == "5d" or (target_format == "auto" and data.ndim == 5):
            return data
        # flatten to 3D
        flat3 = data.reshape(n_sims, n_timesteps, -1)
        if target_format == "3d" or (target_format == "auto"):
            return flat3
        # flatten to 2D
        return flat3.reshape(-1, flat3.shape[-1])

    # If none matched, raise
    raise ValueError(
        f'Data shape {getattr(data, "shape", None)} not compatible with n_sims={n_sims}, n_timesteps={n_timesteps}'
    )

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

def get_config():
    """
    Import and return the config module.

    This function handles the import of the examples config module with proper
    fallbacks so the examples can be run both when `examples` is a package and
    when it's just a directory with `config.py` next to the example scripts.

    Returns:
        module: The config module object.

    Raises:
        ImportError: If config.py doesn't exist or can't be imported.
    """

    # 1) Preferred: examples is a package (examples/__init__.py exists)
    try:
        import examples.config as config

        return config
    except Exception:
        pass

    # 2) If running the example from inside the examples folder, a top-level
    #    `import config` may work (e.g. python MEMS.py when cwd is examples/MEMS)
    try:
        import config as config

        return config
    except Exception:
        pass

    # 3) Fallback: try to locate examples/config.py on disk and import it by path
    import importlib.util
    from pathlib import Path

    candidates = []

    # a) examples/config.py relative to project root (assume repo layout)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "examples" / "config.py")
    except Exception:
        pass

    # b) examples/config.py relative to current working directory
    candidates.append(Path.cwd() / "examples" / "config.py")

    # c) examples/config.py next to this utils file (edge case)
    candidates.append(Path(__file__).resolve().parent.parent / "examples" / "config.py")

    # d) direct config.py in cwd
    candidates.append(Path.cwd() / "config.py")

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate.exists() and candidate.is_file():
            try:
                spec = importlib.util.spec_from_file_location("examples_config", str(candidate))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            except Exception:
                # try next candidate
                continue

    # Nothing worked: raise a helpful error
    raise ImportError(
        "Could not import examples config. Please ensure that there is a file named 'config.py' in the examples/ folder. "
        "You can copy examples/config.py.template to examples/config.py and customize it with the correct data paths and parameters for your setup. "
        "Checked locations: {}".format(
            ", ".join(str(p) for p in candidates)
        )
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
            veni, x0, dx0dt0, dx0ddt0, mean_or_sample="mean"
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
