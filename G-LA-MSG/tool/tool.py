import matplotlib
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

matplotlib.use("Agg")


def generate_z(z_dim, dir, device):
    if dir == "A2B":
        z = torch.randn(1, z_dim * 2) * 0.2
        z[:, :z_dim] += 1
    elif dir == "B2A":
        z = torch.randn(1, z_dim * 2) * 0.2
        z[:, z_dim:] += 1
    z = z.to(device)
    return z


def plot_result(x, plot_path, num):
    folder = os.path.exists(plot_path)
    if not folder:
        os.makedirs(plot_path)
    np.savetxt(
        os.path.join(plot_path, str(num) + ".txt"),
        x[0][0].cpu().detach().numpy(),
    )
    plt.plot(x[0][0].cpu().detach().numpy())
    plt.savefig(os.path.join(plot_path, str(num) + ".png"), dpi=600)
    plt.cla()


def get_interpolation(input, plot_path, num, z_dim, gen, device):
    w_A2B = generate_z(z_dim, "A2B", device)
    w_B2A = generate_z(z_dim, "B2A", device)
    for i in range(0, num + 1, 5):
        w = w_A2B + i / num * (w_B2A - w_A2B)
        output, _, _ = gen(input, w)
        plot_result(output, plot_path, i)
