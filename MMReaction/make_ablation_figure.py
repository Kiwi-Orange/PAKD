#!/usr/bin/env python3
"""Generate ablation comparison figure from saved checkpoints."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

variants = ['full', 'no_hidden', 'no_smoothness', 'no_phase']
names = ['Full', 'No hidden', 'No smoothness', 'No phase']
colors = {'Full': '#1f77b4', 'No hidden': '#ff7f0e', 'No smoothness': '#2ca02c', 'No phase': '#d62728'}

losses_dict = {}
for v, n in zip(variants, names):
    path = f'/Users/ransheng/PythonProjects/PAKD/MMReaction/results/ablation/student_{v}.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    losses_dict[n] = ckpt['training_losses']
    print(f"{n:12s}: {len(losses_dict[n]['total'])} epochs, final loss = {losses_dict[n]['total'][-1]:.4e}")

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 600,
})

fig = plt.figure(figsize=(7.2, 4.8))
grid = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.28)

ax_loss = fig.add_subplot(grid[0, 0])
for n in names:
    epochs = np.arange(1, len(losses_dict[n]['total']) + 1)
    ax_loss.semilogy(epochs, losses_dict[n]['total'], color=colors[n], lw=2.0, label=n)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Total loss")
ax_loss.legend(frameon=False, loc="upper right")
ax_loss.set_title("Training convergence")
ax_loss.grid(True, alpha=0.3, lw=0.5)

ax_bar = fig.add_subplot(grid[0, 1])
vals = [losses_dict[n]['total'][-1] for n in names]
bars = ax_bar.bar(range(len(names)), vals, color=[colors[n] for n in names], edgecolor="black", lw=0.6, width=0.6)
ax_bar.set_xticks(range(len(names)), names, rotation=15, ha="right")
ax_bar.set_ylabel("Final total loss")
ax_bar.set_title("Final loss comparison")
for bar, val in zip(bars, vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{val:.3e}", ha="center", va="bottom", fontsize=8, fontweight="bold")

out = '/Users/ransheng/PythonProjects/PAKD/MMReaction/results/ablation/ablation_comparison'
fig.savefig(f'{out}.png', facecolor="white")
fig.savefig(f'{out}.pdf', facecolor="white")
plt.close(fig)
print(f"\n✓ Figure saved: {out}.png / {out}.pdf")
