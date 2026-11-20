import os
import matplotlib.pyplot as plt
import numpy as np

def generate_fig4():
    # Style settings matching Springer / existing figures
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })

    ranks = [1, 2, 4, 8, 16]
    # Canonical 5-seed values
    accuracies = [93.49, 93.50, 93.65, 93.75, 93.88]
    betas = [1.28, 1.20, 1.10, 1.01, 0.94]

    x_indices = np.arange(len(ranks))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # --- Panel (a): Performance vs. Rank ---
    ax1.plot(x_indices, accuracies, color='#2077A8', linewidth=2.8, marker='o', markersize=9, zorder=3)
    ax1.axhline(93.49, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1) # baseline reference
    
    # Circle the canonical r=4 point
    ax1.scatter(2, accuracies[2], s=250, facecolors='none', edgecolors='red', linewidth=3.0, zorder=4, label='Optimal: $r = 4$')

    ax1.set_title('(a) Performance vs. Rank', fontweight='bold', pad=10)
    ax1.set_xlabel('Rank $r$', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([str(r) for r in ranks])
    ax1.set_ylim(93.2, 94.2)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(loc='lower left', framealpha=0.9)

    # --- Panel (b): Gating vs. Rank ---
    ax2.plot(x_indices, betas, color='#E68A00', linewidth=2.8, marker='s', markersize=9, zorder=3)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.8, label='Hyper-projective threshold', zorder=2)

    # Shaded regions
    ax2.axhspan(1.0, 1.4, color='#d5edd5', alpha=0.8, label='Hyper-projective', zorder=0)
    ax2.axhspan(0.8, 1.0, color='#e0e5ff', alpha=0.8, label='Contractive', zorder=0)

    ax2.set_title('(b) Gating vs. Rank', fontweight='bold', pad=10)
    ax2.set_xlabel('Rank $r$', fontsize=12)
    ax2.set_ylabel(r'Final $\bar{\beta}$', fontsize=12)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.set_ylim(0.8, 1.4)
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()

    out_dirs = [
        "Revision/Tex_Revision/figures",
        "Review_Color/figures",
        "tmp_unzip/Tex_Revision/figures"
    ]
    for od in out_dirs:
        if os.path.exists(od):
            pdf_path = os.path.join(od, "fig4_ablation_rank.pdf")
            png_path = os.path.join(od, "fig4_ablation_rank.png")
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
            print(f"Saved {pdf_path} and {png_path}")

    plt.close()

def generate_fig5():
    groups = [1, 2, 4, 8]
    # Canonical 5-seed values
    accuracies = [93.73, 93.64, 93.65, 93.75]
    params = [897, 919, 941, 963]

    x_indices = np.arange(len(groups))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # --- Panel (a): Performance vs. Groups ---
    ax1.plot(x_indices, accuracies, color='#993366', linewidth=2.8, marker='o', markersize=9, zorder=3)
    ax1.axhline(93.49, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1) # baseline reference
    
    # Circle the G=4 point
    ax1.scatter(2, accuracies[2], s=250, facecolors='none', edgecolors='red', linewidth=3.0, zorder=4, label='Optimal: $G = 4$')

    ax1.set_title('(a) Performance vs. Groups', fontweight='bold', pad=10)
    ax1.set_xlabel('Number of Groups $G$', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([str(g) for g in groups])
    ax1.set_ylim(93.2, 94.2)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(loc='lower right', framealpha=0.9)

    # --- Panel (b): Model Size vs. Groups ---
    bars = ax2.bar(x_indices, params, width=0.35, color='#8fa3a3', edgecolor='black', linewidth=1.2, zorder=2)
    ax2.set_title('(b) Model Size vs. Groups', fontweight='bold', pad=10)
    ax2.set_xlabel('Number of Groups $G$', fontsize=12)
    ax2.set_ylabel('Parameters (K)', fontsize=12)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([str(g) for g in groups])
    ax2.set_ylim(0, 1050)
    ax2.grid(True, alpha=0.3, zorder=0, axis='y')

    for bar, p in zip(bars, params):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{p}K',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    od = "./figures"
    os.makedirs(od, exist_ok=True)
    pdf_path = os.path.join(od, "fig5_ablation_groups.pdf")
    png_path = os.path.join(od, "fig5_ablation_groups.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved {pdf_path} and {png_path}")

    plt.close()

if __name__ == '__main__':
    generate_fig4()
    generate_fig5()
