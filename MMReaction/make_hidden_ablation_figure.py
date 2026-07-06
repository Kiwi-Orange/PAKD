#!/usr/bin/env python3
"""Generate hidden-distillation comparison figure from saved ablation checkpoints."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Only compare: Full PAKD vs No hidden distillation
variants = ['full', 'no_hidden']
names = ['Full PAKD', 'No hidden']
short_names = ['Full', 'No hidden']
colors = {'Full': '#1f77b4', 'No hidden': '#ff7f0e'}

losses_dict = {}
for v, n, sn in zip(variants, names, short_names):
    path = f'/Users/ransheng/PythonProjects/PAKD/MMReaction/results/ablation/student_{v}.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    losses_dict[sn] = ckpt['training_losses']
    print(f"{sn:12s}: {len(losses_dict[sn]['total'])} epochs, final loss = {losses_dict[sn]['total'][-1]:.4e}")

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 600,
})

fig = plt.figure(figsize=(7.2, 3.6))
grid = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.88, bottom=0.16, wspace=0.28)

# --- Left: training loss curves ---
ax_loss = fig.add_subplot(grid[0, 0])
for sn in short_names:
    epochs = np.arange(1, len(losses_dict[sn]['total']) + 1)
    ax_loss.semilogy(epochs, losses_dict[sn]['total'], color=colors[sn], lw=2.2, label=sn)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Total loss")
ax_loss.legend(frameon=False, loc="upper right")
ax_loss.set_title("Hidden-distillation ablation")
ax_loss.grid(True, alpha=0.3, lw=0.5)
for spine in ax_loss.spines.values():
    spine.set_linewidth(0.8)

# --- Right: loss decomposition at final epoch ---
ax_bar = fig.add_subplot(grid[0, 1])
components = ['output', 'hidden', 'smoothness']
full_vals = [losses_dict['Full'][c][-1] for c in components]
no_hidden_vals = [losses_dict['No hidden'][c][-1] for c in components]

x = np.arange(len(components))
width = 0.35
bars1 = ax_bar.bar(x - width/2, full_vals, width, color=colors['Full'], edgecolor='black', lw=0.6, label='Full')
bars2 = ax_bar.bar(x + width/2, no_hidden_vals, width, color=colors['No hidden'], edgecolor='black', lw=0.6, label='No hidden')

ax_bar.set_xticks(x, ['Output', 'Hidden', 'Smoothness'])
ax_bar.set_ylabel("Final loss component")
ax_bar.set_title("Loss decomposition")
ax_bar.legend(frameon=False, loc='upper right')
ax_bar.set_yscale('log')
ax_bar.grid(True, alpha=0.3, lw=0.5, axis='y')
for spine in ax_bar.spines.values():
    spine.set_linewidth(0.8)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, height * 1.15,
                f"{height:.2e}", ha='center', va='bottom', fontsize=7, fontweight='bold')

out = '/Users/ransheng/PythonProjects/PAKD/MMReaction/results/ablation/hidden_ablation_comparison'
fig.savefig(f'{out}.png', facecolor="white")
fig.savefig(f'{out}.pdf', facecolor="white")
plt.close(fig)
print(f"\n✓ Figure saved: {out}.png / {out}.pdf")
