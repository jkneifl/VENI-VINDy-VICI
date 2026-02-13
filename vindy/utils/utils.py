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
    mean_over_epochs, scale_over_epochs, sindy_layer, outdir, model_name, config
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
