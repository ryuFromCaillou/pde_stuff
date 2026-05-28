import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import loadmat


def mat_to_gif(mat_path, out_path_gif, fps=20):
    # --- load the mat file ---
    data = loadmat(mat_path)

    # data keys: 't', 'x', 'u'
    t = np.squeeze(data['t'])   # shape (101,)
    x = np.squeeze(data['x'])   # shape (201,)
    u = data['u']               # shape (101, 201) -> (time, space)

    # --- figure setup ---
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x, u[0, :], lw=2)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.min(u), np.max(u))
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    title = ax.set_title(f't = {t[0]:.3f}')

    def update(frame):
        line.set_ydata(u[frame, :])
        title.set_text(f't = {t[frame]:.3f}')
        return line, title

    # --- animate ---
    interval = 1000 / fps  # ms

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=interval,
        blit=True,
    )

    # --- save to GIF (no ffmpeg needed) ---
    from matplotlib.animation import PillowWriter
    writer_gif = PillowWriter(fps=fps)
    anim.save(out_path_gif, writer=writer_gif)

    print(f'Saved animation to: {out_path_gif}')


file_names = [
    "Allen_Cahn_tanh_0.05.mat",
    "Allen_Cahn_tanh_0.1.mat",
    "Allen_Cahn_tanh_0.2.mat",
    "Allen_Cahn_tanh_0.4.mat",
    "Allen_Cahn_tanh_0.8.mat",
    "Allen_Cahn_tanh_1.0.mat",
    "Allen_Cahn_tanh_2.0.mat",
]

for file_name in file_names:
    mat_path = f"Datasets/data/Diff_Curvature_ghost_pts/{file_name}"
    out_path_gif = f"Datasets/data/Diff_Curvature_ghost_pts/{file_name.replace('.mat', '.gif')}"
    mat_to_gif(mat_path, out_path_gif, fps=20)