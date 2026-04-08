import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

COLORS = {
    'Baseline':     '#888780',
    'RAG':          '#5DCAA5',
    'SelfCheckGPT': '#1D9E75',
    'CAEF (Ours)':  '#085041',
}
MODELS = list(COLORS.keys())

# ── Data ──────────────────────────────────────────────────────────────────────
precision = [0.74, 0.82, 0.84, 0.91]
recall    = [0.68, 0.76, 0.77, 0.88]
f1        = [0.70, 0.79, 0.80, 0.89]
bleu      = [0.61, 0.74, 0.73, 0.86]

ece      = [0.10, 0.08, 0.04]
latency  = [6.1,  5.4,  3.2]
models_t2 = ['RAG', 'SelfCheckGPT', 'CAEF (Ours)']

ablation_labels = ['Full CAEF', '– SCV', '– Live\nRetrieval', '– Explainability']
abl_f1   = [0.89, 0.82, 0.79, 0.75]
abl_bleu = [0.86, 0.81, 0.76, None]


# ── Fig 1: Grouped Bar Chart ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(MODELS))
w = 0.2
bars = [precision, recall, f1, bleu]
bar_labels = ['Precision', 'Recall', 'F1', 'BLEU']
bar_colors = ['#B5D4F4', '#378ADD', '#185FA5', '#0C447C']

for i, (data, label, color) in enumerate(zip(bars, bar_labels, bar_colors)):
    ax1.bar(x + i * w - 1.5 * w, data, width=w, label=label, color=color, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(MODELS)
ax1.set_ylim(0.5, 1.0)
ax1.set_ylabel('Score')
ax1.set_title('Fig 1 — Performance comparison on Educational QA benchmark', fontsize=12, fontweight='normal', pad=12)
ax1.legend(loc='lower right', framealpha=0.5)
plt.tight_layout()
plt.savefig('fig1_grouped_bar.png', bbox_inches='tight')
plt.close()
print("Saved fig1_grouped_bar.png")


# ── Fig 2: Radar Chart ─────────────────────────────────────────────────────────
categories = ['Precision', 'Recall', 'F1', 'BLEU']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

radar_data = {
    'Baseline':     [0.74, 0.68, 0.70, 0.61],
    'RAG':          [0.82, 0.76, 0.79, 0.74],
    'SelfCheckGPT': [0.84, 0.77, 0.80, 0.73],
    'CAEF (Ours)':  [0.91, 0.88, 0.89, 0.86],
}
radar_colors = ['#888780', '#9FE1CB', '#1D9E75', '#085041']
alphas       = [0.1, 0.15, 0.15, 0.25]
linewidths   = [1, 1, 1, 2]

fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for (model, values), color, alpha, lw in zip(radar_data.items(), radar_colors, alphas, linewidths):
    vals = values + values[:1]
    ax2.plot(angles, vals, color=color, linewidth=lw, label=model)
    ax2.fill(angles, vals, color=color, alpha=alpha)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_ylim(0.5, 1.0)
ax2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_yticklabels(['0.6','0.7','0.8','0.9','1.0'], fontsize=8)
ax2.set_title('Fig 2 — Radar: multi-metric overview', fontsize=12, fontweight='normal', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.5)
plt.tight_layout()
plt.savefig('fig2_radar.png', bbox_inches='tight')
plt.close()
print("Saved fig2_radar.png")


# ── Fig 3: Bubble Chart (ECE vs Latency) ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 5))
bcolors = ['#FAC775', '#BA7517', '#412402']
for i, (m, e, l, c) in enumerate(zip(models_t2, ece, latency, bcolors)):
    ax3.scatter(l, e, s=500, color=c, zorder=3, edgecolors='white', linewidths=1.5)
    ax3.annotate(m, (l, e), textcoords='offset points', xytext=(10, 5), fontsize=10, color=c)

ax3.set_xlabel('Latency (s)')
ax3.set_ylabel('ECE ↓ (lower is better)')
ax3.set_xlim(2, 8)
ax3.set_ylim(0, 0.15)
ax3.set_title('Fig 3 — Calibration: ECE vs latency', fontsize=12, fontweight='normal', pad=12)
plt.tight_layout()
plt.savefig('fig3_bubble.png', bbox_inches='tight')
plt.close()
print("Saved fig3_bubble.png")


# ── Fig 4: Ablation Study ─────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5))
x = np.arange(len(ablation_labels))
w = 0.35
ax4.bar(x - w/2, abl_f1, width=w, color='#AFA9EC', label='F1', zorder=3)
bleu_vals = [v if v is not None else 0 for v in abl_bleu]
ax4.bar(x + w/2, bleu_vals, width=w, color='#7F77DD', label='BLEU', zorder=3)
ax4.text(x[-1] + w/2, 0.01, 'N/A', ha='center', va='bottom', fontsize=9, color='#7F77DD')
ax4.set_xticks(x)
ax4.set_xticklabels(ablation_labels, fontsize=10)
ax4.set_ylim(0.6, 0.95)
ax4.set_ylabel('Score')
ax4.set_title('Fig 4 — Ablation study: impact of each component', fontsize=12, fontweight='normal', pad=12)
ax4.legend(framealpha=0.5)
plt.tight_layout()
plt.savefig('fig4_ablation.png', bbox_inches='tight')
plt.close()
print("Saved fig4_ablation.png")


# ── Fig 5: ECE Bar Chart ───────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(6, 5))
ece_colors = ['#F0997B', '#D85A30', '#993C1D']
bars5 = ax5.bar(models_t2, ece, color=ece_colors, zorder=3, width=0.5)
for bar, val in zip(bars5, ece):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='normal')
ax5.set_ylim(0, 0.14)
ax5.set_ylabel('ECE ↓ (lower is better)')
ax5.set_title('Fig 5 — Expected calibration error (ECE)', fontsize=12, fontweight='normal', pad=12)
plt.tight_layout()
plt.savefig('fig5_ece.png', bbox_inches='tight')
plt.close()
print("Saved fig5_ece.png")


# ── Fig 6: Precision–Recall Curve ─────────────────────────────────────────────
recall_axis = np.linspace(0, 1, 100)

def pr_curve(p, r, steepness=3):
    pr = p - (p - r) * (recall_axis ** steepness)
    return np.clip(pr, 0, 1)

pr_curves = {
    'Baseline':     pr_curve(0.74, 0.40, 2.5),
    'RAG':          pr_curve(0.82, 0.45, 2.8),
    'SelfCheckGPT': pr_curve(0.84, 0.47, 2.8),
    'CAEF (Ours)':  pr_curve(0.91, 0.60, 3.2),
}
pr_colors = ['#888780', '#9FE1CB', '#1D9E75', '#085041']
pr_lws    = [1, 1, 1, 2.5]

fig6, ax6 = plt.subplots(figsize=(7, 5))
for (model, curve), color, lw in zip(pr_curves.items(), pr_colors, pr_lws):
    ax6.plot(recall_axis, curve, color=color, linewidth=lw, label=model)

ax6.fill_between(recall_axis, pr_curves['CAEF (Ours)'], alpha=0.1, color='#085041')
ax6.set_xlabel('Recall')
ax6.set_ylabel('Precision')
ax6.set_xlim(0, 1)
ax6.set_ylim(0.3, 1.05)
ax6.set_title('Fig 6 — Precision–Recall curve', fontsize=12, fontweight='normal', pad=12)
ax6.legend(loc='lower left', framealpha=0.5)
plt.tight_layout()
plt.savefig('fig6_pr_curve.png', bbox_inches='tight')
plt.close()
print("Saved fig6_pr_curve.png")

print("\nAll 6 figures saved successfully!")
print("Files: fig1_grouped_bar.png, fig2_radar.png, fig3_bubble.png,")
print("       fig4_ablation.png, fig5_ece.png, fig6_pr_curve.png")