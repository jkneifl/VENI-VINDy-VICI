import numpy as np
import os
import matplotlib.pyplot as plt
import imageio


def add_lognormal_noise(trajectory, sigma):
    """
    Add lognormal noise to a trajectory.

    Parameters
    ----------
    trajectory : array-like
        The trajectory data to add noise to.
    sigma : float
        Standard deviation parameter for the lognormal distribution.

    Returns
    -------
    tuple of array-like
        Tuple containing (noisy_trajectory, noise).
    """
    noise = np.random.lognormal(mean=0, sigma=sigma, size=trajectory.shape)
    return trajectory * noise, noise


def coefficient_distributions_to_csv(sindy_layer, outdir, var_names=[], param_names=[]):
    """
    Save the coefficient distributions of the SINDy layer to CSV files.

    Parameters
    ----------
    sindy_layer : SindyLayer
        SINDy layer containing coefficient distributions.
    outdir : str
        Output directory for CSV files.
    var_names : list of str, optional
        Names of state variables (default is []).
    param_names : list of str, optional
        Names of parameters (default is []).
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
    scale = sindy_layer.priors.reverse_log(log_scale.detach().cpu().numpy())

    mean_values, scale_values = (
        mean.detach().cpu().numpy().reshape(n_vars, n_features).T,
        scale.reshape(n_vars, n_features).T if isinstance(scale, np.ndarray) else scale.detach().cpu().numpy().reshape(n_vars, n_features).T,
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
    Create a GIF showing how coefficient distributions evolve over epochs.

    Parameters
    ----------
    mean_over_epochs : list of array-like
        Mean values of coefficients at each epoch.
    scale_over_epochs : list of array-like
        Scale values of coefficients at each epoch.
    sindy_layer : SindyLayer
        SINDy layer instance with visualization methods.
    outdir : str
        Output directory for saving frames and the GIF.
    """
    os.makedirs(os.path.join(outdir, "coefficients"), exist_ok=True)
    # determine how many frames to generate (capped at 401 frames: indices 0-400)
    max_frames = min(len(mean_over_epochs), len(scale_over_epochs), 401)
    # create gif showing how the coefficient distributions evolve over time
    for i, (mean_, scale_) in enumerate(zip(mean_over_epochs, scale_over_epochs)):
        if i >= max_frames:
            break
        x_range = 1.5
        fig = sindy_layer._visualize_coefficients(
            mean_, scale_, x_range=[-x_range, x_range], y_range=[0, 6]
        )
        fig.suptitle(f"Epoch {i}")
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
    Plot the training history.

    Parameters
    ----------
    history : dict
        Training history object or dictionary.
    outdir : str
        Output directory path for saving plots.
    validation : bool, default=True
        Whether to plot validation metrics.
    """
    os.makedirs(outdir, exist_ok=True)
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
    Plot the coefficient training history.

    Parameters
    ----------
    history : dict
        Training history containing 'coeffs_mean' and 'coeffs_scale'.
    outdir : str
        Output directory path for saving plots.
    """
    mean_over_epochs = np.array(history["coeffs_mean"]).squeeze()
    scale_over_epochs = np.array(history["coeffs_scale"]).squeeze()
    os.makedirs(outdir, exist_ok=True)
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
    - spatial_shape: optional tuple describing spatial dims.
    - target_format: 'auto' (default), '2d', '3d', or '5d'.

    Returns
    - Converted np.ndarray in requested format.
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
        flat3 = data.reshape(n_sims, n_timesteps, -1)
        if target_format == "3d" or (target_format == "auto"):
            return flat3
        return flat3.reshape(-1, flat3.shape[-1])

    raise ValueError(
        f'Data shape {getattr(data, "shape", None)} not compatible with n_sims={n_sims}, n_timesteps={n_timesteps}'
    )

"""
Shared utility functions for examples.
"""

import os
import random
import logging
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_config():
    """
    Import and return the config module.

    Returns
    -------
    module
        The config module object.

    Raises
    ------
    ImportError
        If config.py doesn't exist or can't be imported.
    """

    try:
        import examples.config as config
        return config
    except Exception:
        pass

    try:
        import config as config
        return config
    except Exception:
        pass

    import importlib.util
    from pathlib import Path

    candidates = []
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "examples" / "config.py")
    except Exception:
        pass
    candidates.append(Path.cwd() / "examples" / "config.py")
    candidates.append(Path(__file__).resolve().parent.parent / "examples" / "config.py")
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
                continue

    raise ImportError(
        "Could not import examples config. Please ensure that there is a file named 'config.py' in the examples/ folder. "
        "You can copy examples/config.py.template to examples/config.py and customize it with the correct data paths and parameters for your setup. "
        "Checked locations: {}".format(
            ", ".join(str(p) for p in candidates)
        )
    )


def set_seed(seed: int):
    """
    Set seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_data_path(data_path: str, zenodo_doi: str = "10.5281/zenodo.18313843"):
    """
    Validate that a data file exists and provide a helpful error message if not.

    Parameters
    ----------
    data_path : str
        Path to the data file.
    zenodo_doi : str, optional
        Zenodo DOI for downloading the data.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. "
            f"Please download the file from Zenodo (http://doi.org/{zenodo_doi}) and "
            f"specify the correct path in the examples/config.py file."
        )


def log_model_summary(model, result_dir: str = None):
    """
    Log a summary of the model architecture.

    Parameters
    ----------
    model : nn.Module
        The model instance.
    result_dir : str, optional
        Directory to save the summary text file.
    """
    try:
        logging.info("=" * 60)
        logging.info("Model Summary:")
        logging.info("=" * 60)

        if hasattr(model, 'reduced_order'):
            logging.info(f"Reduced order: {model.reduced_order}")
        if hasattr(model, 'second_order'):
            logging.info(f"Second order: {model.second_order}")
        if hasattr(model, 'scaling'):
            logging.info(f"Scaling method: {model.scaling}")

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        summary_lines = [
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
        ]

        # Save to file if directory provided
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            summary_path = os.path.join(result_dir, "model_summary.txt")
            with open(summary_path, "w") as f:
                f.write(str(model))
                f.write("\n\n")
                f.write("\n".join(summary_lines))
            logging.info(f"Model summary saved to {summary_path}")

        logging.info("=" * 60)
    except Exception as e:
        logging.warning(f"Failed to log model summary: {e}")


def get_latent_initial_conditions(veni, x, dxdt, dxddt, mean_or_sample):
    """
    Compute the initial conditions in latent space for integration.

    Parameters
    ----------
    veni : model
        The model instance.
    x : array-like
        The state data array.
    dxdt : array-like
        The first time derivative.
    dxddt : array-like or None
        The second time derivative.
    mean_or_sample : str
        Whether to compute mean or sample ("mean" or "sample").

    Returns
    -------
    array-like
        Initial conditions in latent space.
    """
    if veni.second_order:
        z0, dz0, _ = veni.calc_latent_time_derivatives(
            x, dxdt, dxddt, mean_or_sample=mean_or_sample
        )
        ic = np.concatenate([z0[0], dz0[0]], axis=-1)
    else:
        z0, dz0 = veni.calc_latent_time_derivatives(
            x, dxdt, None, mean_or_sample=mean_or_sample
        )
        ic = z0[0]

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
    Perform inference on test trajectories.

    Parameters
    ----------
    veni : model
        The trained model.
    sim_ids : list of int
        Test trajectory indices.
    n_sims : int
        Number of simulations.
    n_timesteps : int
        Number of timesteps.
    t : array-like
        Test time steps.
    x : array-like
        Scaled test data.
    dxdt : array-like, optional
        Test data derivatives.
    params : array-like, optional
        Test parameters.

    Returns
    -------
    tuple
        (Z, z_preds, t_preds)
    """
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

    z_preds = np.array(z_preds)
    t_preds = np.array(t_preds)

    return Z, z_preds, t_preds


def plot_inference_results(t_preds, z_preds, T, Z, sim_ids, state_id=0):
    """
    Plot inference results for test trajectories.

    Parameters
    ----------
    t_preds : list of array-like
        Predicted time steps.
    z_preds : list of array-like
        Predicted latent states.
    T : array-like
        True time steps.
    Z : array-like
        True latent states.
    sim_ids : list of int
        Simulation indices to plot.
    state_id : int, optional
        State variable index to plot.
    """
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
    Perform forward uncertainty quantification by sampling trajectories.

    Parameters
    ----------
    veni : model
        The model instance.
    sim_ids : list of int
        Simulation indices.
    n_traj : int
        Number of trajectories to sample.
    n_sims : int
        Total number of simulations.
    n_timesteps : int
        Timesteps per simulation.
    t : array-like
        Time vector.
    x : array-like
        State data.
    dxdt : array-like
        First time derivative.
    dxddt : array-like, optional
        Second time derivative.
    params : array-like, optional
        Additional parameters.
    sigma : float, optional
        Number of standard deviations for confidence intervals.

    Returns
    -------
    dict
        Dictionary with UQ results.
    """

    second_order = veni.second_order

    def _to_numpy(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    x = _to_numpy(x)
    dxdt = _to_numpy(dxdt)
    dxddt = _to_numpy(dxddt)
    params = _to_numpy(params)
    t = _to_numpy(t)

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

    if second_order:
        z, dzdt, _ = veni.calc_latent_time_derivatives(x, dxdt, dxddt)
    else:
        z, dzdt = veni.calc_latent_time_derivatives(x, dxdt, None)

    Z = switch_data_format(_to_numpy(z), n_sims, n_timesteps, target_format="3d")
    DZDT = switch_data_format(_to_numpy(dzdt), n_sims, n_timesteps, target_format="3d")

    # Save kernel state
    kernel_orig = veni.sindy_layer.kernel.data.clone()
    kernel_scale_orig = veni.sindy_layer.kernel_scale.data.clone()

    sampled_times = []
    latent_trajectories_samples = []
    mean_latent_trajectories = []

    for i in sim_ids:
        logging.info("Processing trajectory %d/%d", i + 1, len(sim_ids))

        mu = Params[i] if params is not None else None
        tvec = T[i].squeeze()

        x0, dx0dt0, dx0ddt0 = (
            X[i, 0:1],
            DXDT[i, 0:1],
            DXDDT[i, 0:1] if second_order else None,
        )

        traj_samples = []
        traj_times = []
        for traj in range(n_traj):
            logging.info("\tSampling model %d/%d", traj + 1, n_traj)

            ic = get_latent_initial_conditions(
                veni, x0, dx0dt0, dx0ddt0, mean_or_sample="sample"
            )

            sol, coeffs = veni.sindy_layer.integrate_uq(ic, tvec, mu=mu)

            traj_samples.append(np.asarray(sol.y))
            traj_times.append(np.asarray(sol.t))

        sampled_times.append(traj_times)
        latent_trajectories_samples.append(traj_samples)

        # Restore kernel state for mean integration
        veni.sindy_layer.kernel.data.copy_(kernel_orig)
        veni.sindy_layer.kernel_scale.data.copy_(kernel_scale_orig)

        ic = get_latent_initial_conditions(
            veni, x0, dx0dt0, dx0ddt0, mean_or_sample="mean"
        )

        sol_mean = veni.integrate(ic, tvec, mu=mu)
        mean_latent_trajectories.append(np.asarray(sol_mean.y))

    latent_trajectories_samples = np.array(latent_trajectories_samples)
    latent_trajectories_samples = np.transpose(
        latent_trajectories_samples, (0, 1, 3, 2)
    )

    mean_latent_samples = np.mean(latent_trajectories_samples, axis=1)
    std_latent_samples = np.std(latent_trajectories_samples, axis=1)

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
    Generate uncertainty quantification plots.

    Parameters
    ----------
    sampled_times : list
        Time points for sampled UQ trajectories.
    mean_latent : array-like
        Mean trajectories.
    mean_latent_samples : array-like
        Mean of sampled trajectories.
    std_latent_samples : array-like
        Standard deviation of sampled trajectories.
    t_test : array-like
        Test time steps.
    z_test : array-like
        Latent states for test data.
    test_ids : list of int
        Test trajectory indices.
    state_id : int, optional
        State variable index to plot.
    """
    n_test = len(test_ids)
    fig, axs = plt.subplots(n_test, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Integrated Test Trajectories")
    for i, i_test in enumerate(test_ids):
        axs[i].set_title(f"Test Trajectory {i_test + 1}")
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
