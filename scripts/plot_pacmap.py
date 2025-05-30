import fireducks.pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scienceplots

# Import parquet file
df = pd.read_parquet('data_test/test_pacmap.parquet')

# Prepare Data to Plot
x = df['pacmap1']
y = df['pacmap2']

# Plot params
pparam = dict(
    xlabel = 'PACMAP 1',
    ylabel = 'PACMAP 2',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.scatter(x, y, s=2, color='silver', alpha=0.3, label='_nolegend_')
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    x_padding = (xmax - xmin) * 0.1
    y_padding = (ymax - ymin) * 0.1
    grid_xmin = xmin - x_padding
    grid_xmax = xmax + x_padding
    grid_ymin = ymin - y_padding
    grid_ymax = ymax + y_padding

    xx, yy = np.mgrid[grid_xmin:grid_xmax:100j, grid_ymin:grid_ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    try:
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Contour plot
        contour_plot = ax.contourf(xx, yy, f, levels=10, cmap='Blues', alpha=0.6)
        cbar = fig.colorbar(contour_plot, ax=ax, label='Density', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
    except Exception as e:
        print(f"Error in KDE computation: {e}")

    ax.legend()
    fig.savefig('figs/pacmap_density_plot.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
