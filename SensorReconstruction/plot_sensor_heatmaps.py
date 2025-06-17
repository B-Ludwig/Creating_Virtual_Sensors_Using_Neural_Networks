import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LogNorm

# Set Times New Roman as the global font

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_comparison(main_folder1, main_folder2,
                    output_path=None, cmap_name='viridis', save_pdf=False):
    """
    Plots two heatmaps (classic vs masked) of reconstruction MSE stacked vertically,
    sharing x-axis labels only on the bottom plot, with a common caption above.

    Parameters
    ----------
    main_folder1, main_folder2 : str
        Paths to the two dataset folders containing cluster subfolders.
    output_path : str, optional
        If provided, folder in which to save outputs.
    cmap_name : str, optional
        Matplotlib colormap for both heatmaps.
    save_pdf : bool
        If True, also saves figure as PDF.
    """

    # helper: prettify sensor names into multi-line labels
    def prettify_sensor(name):
        dev = name.split('.')[0]
        b_match = re.search(r"B\d+", name)
        loc = b_match.group(0) if b_match else ''
        typ = name.split('.')[-1]
        type_map = {
            'p': 'Pressure',
            'T': 'Temp',
            'level': 'Level',
            'N_in': 'Speed',
            'n_in': 'Speed',
            'V_flow': 'VolFlow',
            'm_flow': 'MassFlow',
        }
        t = type_map.get(typ, typ)
        return f"{dev}_{loc}\n{t}" if loc else f"{dev}\n{t}"

    # identify cluster subdirectories
    subdirs = [d for d in os.listdir(main_folder1)
               if os.path.isdir(os.path.join(main_folder1, d))]
    def cluster_key(name):
        if name.lower() == 'full': return float('inf')
        m = re.match(r'.*?(\d+)$', name)
        return int(m.group(1)) if m else name
    clusters = sorted(subdirs, key=cluster_key)

    # create display labels for clusters: numeric or Full
    display_clusters = []
    for cl in clusters:
        if cl.lower() == 'full':
            display_clusters.append('Full')
        else:
            m = re.match(r'.*?(\d+)$', cl)
            display_clusters.append(m.group(1) if m else cl)

    # load sensor columns per cluster
    cluster_sensors = {}
    for cluster in clusters:
        jf = os.path.join(main_folder1, cluster, 'best_hparams.json')
        if not os.path.exists(jf): raise FileNotFoundError(jf)
        with open(jf) as f:
            params = json.load(f)
        cluster_sensors[cluster] = params['experiments'][0]['DATA']['COLUMNS']

    # ordered unique list of sensors
    all_sensors = []
    seen = set()
    for cluster in clusters:
        for s in cluster_sensors[cluster]:
            if s not in seen:
                seen.add(s)
                all_sensors.append(s)

    # helper to build DataFrame of MSE values
    def build_mse_df(folder):
        dfm = pd.DataFrame(index=clusters, columns=all_sensors, dtype=float)
        for cluster in clusters:
            path = os.path.join(folder, cluster)
            files = glob.glob(os.path.join(path, 'loss_summary_reconstruction*.csv'))
            if not files: raise FileNotFoundError(path)
            df = pd.read_csv(files[0])
            names = cluster_sensors[cluster]
            for _, row in df.iterrows():
                idx = int(row['sensor_index'])
                if 0 <= idx < len(names):
                    dfm.at[cluster, names[idx]] = row['mean_loss_x_only_recon_gt']
        return dfm

    df1 = build_mse_df(main_folder1).values.astype(float)
    df2 = build_mse_df(main_folder2).values.astype(float)
    df1[df1 > 1] = 1
    df2[df2 > 1] = 1

    # combine for log normalization
    pos = np.concatenate([df1[(df1 > 0) & np.isfinite(df1)],
                          df2[(df2 > 0) & np.isfinite(df2)]])
    if not pos.size: raise ValueError("No positive values.")
    norm = LogNorm(vmin=pos.min(), vmax=pos.max())

    # prepare figure
    width = max(32, len(all_sensors) * 0.5)
    height = max(6, len(clusters) * 0.6)
    fig, axes = plt.subplots(2, 1, figsize=(width, height * 2), sharex=True)
    # after creating fig, axes …
    left, right = fig.subplotpars.left, fig.subplotpars.right
    mid = 1 - (left + right) / 2
    print(f'mid: {mid}')
    fig.suptitle(
        'Reconstruction of missing sensor values',
        fontsize=24,
        x=mid,   # center over subplot area
        y=0.98
    )

    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad('lightgray')

    # prettified sensor labels
    pretty = [prettify_sensor(s) for s in all_sensors]

    for ax, data, title in zip(axes, [df1, df2], ['Classic Training', 'Masked Training']):
        cax = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)
        ax.set_yticks(np.arange(len(clusters)))
        ax.set_yticklabels(display_clusters, fontsize=20)
        ax.set_title(title, fontsize=22)
        if ax is axes[0]:
            ax.set_xticks([])
        else:
            ax.set_xticks(np.arange(len(all_sensors)))
            ax.set_xticklabels(pretty, rotation=90, fontsize=20)

    axes[-1].set_xlabel('Masked Sensor', fontsize=22)
    axes[0].set_ylabel('Cluster', fontsize=22)
    axes[1].set_ylabel('Cluster', fontsize=22)

    # colorbar
    fig.subplots_adjust(right=0.93, top=0.98)

    # get the bounding boxes of the two heatmap axes
    pos0 = axes[0].get_position()   # Bbox(x0, y0, x1, y1) of top subplot
    pos1 = axes[1].get_position()   # Bbox of bottom subplot
    # set bottom at bottom of lower plot (pos1.y0)
    # height from bottom of lower to top of upper (pos0.y1 - pos1.y0)
    cbar_ax = fig.add_axes([
        0.94,                                               # x-start (just to the right)
        pos1.y0 + (pos0.y1 - pos1.y0) / (2 / 1/(1-0.55)),   # y-start at bottom of 2nd plot
        0.02,                                               # width of colorbar
        (pos0.y1 - pos1.y0) * 0.6                           # height spanning both plots
    ])
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Mean Loss (MSE)', fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(rect=[0, 0, 0.93, 0.98])

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        png = os.path.join(output_path, 'comparison_heatmap.png')
        fig.savefig(png, dpi=300)
        if save_pdf:
            pdf = os.path.join(output_path, 'comparison_heatmap.pdf')
            fig.savefig(pdf, format='pdf', bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('folder1')
    p.add_argument('folder2')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--cmap', default='viridis')
    p.add_argument('--pdf', action='store_true')
    args = p.parse_args()
    plot_comparison(args.folder1, args.folder2,
                    output_path=args.output,
                    cmap_name=args.cmap,
                    save_pdf=args.pdf)
