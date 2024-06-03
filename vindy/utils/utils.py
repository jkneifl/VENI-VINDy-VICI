import numpy as np
import os


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

