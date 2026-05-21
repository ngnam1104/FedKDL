import matplotlib.pyplot as plt

COLORS = {
    'fedavg': '#E63946',
    'fedprox': '#F4A261',
    'hfl_nocoop': '#2A9D8F',
    'hfl_selective': '#264653',
    'hfl_nearest': '#8AB17D'
}

MARKERS = {
    'fedavg': 'o',
    'fedprox': 's',
    'hfl_nocoop': '^',
    'hfl_selective': 'D',
    'hfl_nearest': 'v'
}

LABELS = {
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'hfl_nocoop': 'HFL-NoCoop',
    'hfl_selective': 'HFL-Selective (Ours)',
    'hfl_nearest': 'HFL-Nearest'
}

def get_style(baseline: str):
    return (
        COLORS.get(baseline, '#000000'),
        MARKERS.get(baseline, 'x'),
        LABELS.get(baseline, baseline)
    )

def setup_global_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.figsize': (8, 6),
        'figure.dpi': 150
    })
