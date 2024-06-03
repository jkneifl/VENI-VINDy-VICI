import os
import scipy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from vindy.utils import add_lognormal_noise


def data_generation(
    ode,
    n_train,
    n_test,
    random_IC,
    random_a,
    seed,
    model_noise_factor,
    measurement_noise_factor,
):

    # time vectore
    t0, T, nt = 0, 24, 2000
    t = np.linspace(t0, T, nt)

    # Roessler system parameters
    a = 0.2
    b = 0.2
    c = 5.7
    # initial conditions
    x0 = -5
    y0 = -5
    z0 = 0
    ic = [x0, y0, z0]
    dim = 3
    var_names = ["z_1", "z_2", "z_3"]

    # Generate initial conditions and parameters
    if random_IC:
        np.random.seed(seed)
        x0 = np.concatenate(
            [np.random.normal(ic_, scale=2, size=(n_train + n_test, 1)) for ic_ in ic],
            axis=1,
        )
    else:
        x0 = np.repeat(np.array([ic])[:, :, np.newaxis], n_train + 1, axis=2)

    if random_a:
        np.random.seed(seed)
        a_samples = np.random.normal(a, a * model_noise_factor, size=n_train)
        b_samples = np.random.normal(b, b * model_noise_factor, size=n_train)
        c_samples = np.random.normal(c, c * model_noise_factor, size=n_train)
    else:
        a_samples = [a]
        b_samples = [b]
        c_samples = [c]

    # Generate data
    x = np.array(
        [
            scipy.integrate.odeint(lambda x, t: ode(t, x, a=a_, b=b_, c=c_), x0_, t)
            for i, (a_, b_, c_, x0_) in enumerate(
                zip(a_samples, b_samples, c_samples, x0[:n_train])
            )
        ]
    )

    # add measurement noise
    x = np.array([add_lognormal_noise(x_, measurement_noise_factor)[0] for x_ in x])
    x_test = np.array(
        [
            scipy.integrate.odeint(lambda x, t: ode(t, x, a=a, b=b, c=c), x0_, t)
            for x0_ in x0[n_train:]
        ]
    )

    # calculate time derivatives
    dxdt = [np.array(np.gradient(x_, t[1] - t[0], axis=0)) for x_ in x]
    dxdt_test = [np.array(np.gradient(x_, t[1] - t[0], axis=0)) for x_ in x_test]

    return (
        t,
        x,
        dxdt,
        x_test,
        dxdt_test,
        var_names,
        dim,
    )


def generate_directories(model_name, sindy_type, scenario_info, outdir):
    # noise before derivative, model error, seed
    outdir = os.path.join(outdir, f"{model_name}/", f"{sindy_type}/")
    figdir = os.path.join(outdir, f"figures/{scenario_info}")
    log_dir = os.path.join(
        outdir,
        f'{model_name}/log/{scenario_info}_{datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")}',
    )
    weights_dir = os.path.join(outdir, f"weights/{scenario_info}")
    # save figure
    for dir in [outdir, figdir, log_dir, weights_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    return outdir, figdir, log_dir, weights_dir


def data_plot(t, x, dxdt, x_test):
    # three-dimensional plot of roessler attractor
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, x_ in enumerate(x):
        if i == 0:
            ax.plot(x_[:, 0], x_[:, 1], x_[:, 2], c="gray", label="Training data")
        else:
            ax.plot(x_[:, 0], x_[:, 1], x_[:, 2], c="gray")
    for i, x_ in enumerate(x_test):
        if i == 0:
            ax.plot(x_[:, 0], x_[:, 1], x_[:, 2], c="k", label="Test data")
        else:
            ax.plot(x_[:, 0], x_[:, 1], x_[:, 2], c="k")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    ax.set_zlabel("$z_3$")
    plt.legend()
    plt.tight_layout()

    # # Plot the training data
    fix, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(len(x)):
        axs[0].plot(t, x[i][:, 0], "b")
        axs[0].plot(t, x[i][:, 1], "r")
        axs[1].plot(t, dxdt[i][:, 0], "b")
        axs[1].plot(t, dxdt[i][:, 1], "r")
    axs[0].set_xlabel("t")
    axs[0].legend(["x", "y"], fontsize=14)
    axs[0].set_title("States")
    axs[1].set_xlabel("t")
    axs[1].legend(["$\dot{x}$", "$\dot{y}$"], fontsize=14)
    axs[1].set_title("Velocities")


def training_plot(sindy_layer, trainhist, var_names):
    plt.figure()
    plt.title("Loss over epochs")
    plt.semilogy(trainhist.history["loss"])
    plt.semilogy(trainhist.history["dz"])
    plt.semilogy(trainhist.history["kl_sindy"])
    plt.legend(["loss", "dz", "kl_sindy"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.title("VINDy Coefficients over epochs")
    plt.plot(np.array(trainhist.history["coeffs_mean"]).squeeze())
    plt.xlabel("Epoch")
    plt.ylabel("Coefficient")
    plt.show()

    equation = sindy_layer.model_equation_to_str(z=var_names, precision=3)
    sindy_layer.visualize_coefficients(x_range=[-1.6, 1.6], z=var_names, mu=None)
    plt.suptitle(equation)
    plt.tight_layout()


def trajectory_plot(t, x_test, t_pred, x_pred, dim, nt, i_test, var_names):
    fig, axs = plt.subplots(dim, 1, figsize=(12, 4))
    fig.suptitle(f"Integrated Test Trajectory")
    t_0 = i_test * int(nt)
    for i in range(dim):
        axs[i].plot(
            t, x_test[t_0 : t_0 + nt, i], label=f"${var_names[i]}$", color="black"
        )
        axs[i].plot(
            t_pred,
            x_pred[i],
            label=f"${var_names[i]}^s$",
            linestyle="--",
            color="orange",
        )
        axs[i].set_xlabel("$t$")
    plt.legend()
    plt.tight_layout()

def uq_plot(t, x_test, t_preds, x_pred, x_uq_mean_sampled, x_uq_std, dim, nt, i_test):
    fig, axs = plt.subplots(dim, 1, figsize=(12, 4), sharex=True)
    fig.suptitle(f"Integrated Test Trajectories")
    t_0 = i_test * int(nt)
    axs[0].set_title(f"Test Trajectory {i_test + 1}")
    for i in range(dim):
        axs[i].fill_between(
            t_preds[i],
            x_uq_mean_sampled[i] - 3 * x_uq_std[i],
            x_uq_mean_sampled[i] + 3 * x_uq_std[i],
            color="grey",
            alpha=0.3,
            label="uncertainty bound (+-3 std)",
        )
        axs[i].plot(t, x_test[t_0: t_0 + nt, i], color="black", label="reference")
        axs[i].plot(t_preds[i], x_pred[i], color="orange", linestyle="--", label="mean prediction")
    plt.legend()
    plt.tight_layout()