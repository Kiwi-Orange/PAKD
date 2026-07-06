#!/usr/bin/env python3
"""Test script for the new 'Accuracy vs condition difficulty' panel."""

import sys
sys.path.insert(0, '/Users/ransheng/PythonProjects/PAKD/MMReaction')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_nature_figure import (
    generate_test_conditions,
    compute_all_trajectories,
    load_checkpoint_model,
    TEACHER_MODEL,
    STUDENT_MODEL,
    TIME_POINTS,
    SPECIES,
    COLORS,
    configure_style,
    style_axis,
    add_panel_label,
)

configure_style()

# Load models
print("Loading checkpoints...")
teacher_bundle = load_checkpoint_model(TEACHER_MODEL, is_student=False)
student_bundle = load_checkpoint_model(STUDENT_MODEL, is_student=True)

# Generate conditions and trajectories
print("Generating trajectories...")
conditions = generate_test_conditions(50)
trajectories = compute_all_trajectories(teacher_bundle, student_bundle, conditions)

# Compute RMSE per condition
truth = trajectories["analytical"]
rmse = {}
for method in ["qssa", "teacher", "student"]:
    err = trajectories[method] - truth
    # RMSE across time and species for each condition
    rmse[method] = np.sqrt(np.mean(err**2, axis=(1, 2)))

# Condition difficulty metric: log10(S0/E0)
log_ratio = np.log10(conditions[:, 1] / conditions[:, 0])

# Sort by difficulty for plotting
order = np.argsort(log_ratio)
log_ratio_sorted = log_ratio[order]

# QSSA failure zone: where |log(S0/E0)| < 1, i.e., S0/E0 between 0.1 and 10
# More precisely, QSSA works best when S0 >> E0 (log_ratio >> 0) or E0 >> S0 (log_ratio << 0)
# The problematic zone is around log_ratio ≈ 0
failure_lo = -0.5
failure_hi = 0.5

fig, ax = plt.subplots(figsize=(5.5, 4.2))

# Shade QSSA failure zone
ax.axvspan(failure_lo, failure_hi, color="#FEF3C7", alpha=0.65, zorder=0)
ax.text(
    0.0, 0.95, "QSSA failure zone",
    transform=ax.transAxes, ha="center", va="top",
    fontsize=6.5, color="#92400E", fontweight="bold",
    bbox=dict(facecolor="#FEF3C7", edgecolor="none", alpha=0.8, pad=0.4),
)

# Plot RMSE curves
styles = {
    "qssa": dict(color=COLORS["qssa"], lw=2.0, ls=":", marker="o", ms=3.5, mfc="white", mec=COLORS["qssa"], mew=0.8),
    "teacher": dict(color=COLORS["teacher"], lw=2.0, ls="--", marker="s", ms=3.5, mfc="white", mec=COLORS["teacher"], mew=0.8),
    "student": dict(color=COLORS["student"], lw=2.3, ls="-", marker="D", ms=3.5, mfc=COLORS["student"], mec=COLORS["student"], mew=0.8),
}
labels = {
    "qssa": "QSSA",
    "teacher": "Teacher",
    "student": "PAKD student",
}

for method in ["qssa", "teacher", "student"]:
    y = rmse[method][order]
    ax.semilogy(log_ratio_sorted, y, label=labels[method], **styles[method])

ax.set_xlabel(r"$\log_{10}(S_0/E_0)$", labelpad=1.5)
ax.set_ylabel("RMSE", labelpad=1.5)
ax.set_title("Error vs condition difficulty", pad=2)
ax.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.2, borderpad=0.0)
style_axis(ax, grid=True)

# Add annotation for PAKD advantage in failure zone
mask_fail = (log_ratio >= failure_lo) & (log_ratio <= failure_hi)
if mask_fail.any():
    qssa_fail_mean = np.mean(rmse["qssa"][mask_fail])
    student_fail_mean = np.mean(rmse["student"][mask_fail])
    ax.text(
        0.98, 0.15,
        f"In failure zone:\n"
        f"QSSA RMSE = {qssa_fail_mean:.2e}\n"
        f"Student RMSE = {student_fail_mean:.2e}\n"
        f"Improvement: {qssa_fail_mean/student_fail_mean:.1f}×",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.8, color=COLORS["student"], fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
    )

plt.tight_layout()
out_path = "/Users/ransheng/PythonProjects/PAKD/MMReaction/results/nature_figure/test_accuracy_panel.png"
plt.savefig(out_path, dpi=300, facecolor="white")
print(f"Saved test panel to {out_path}")

# Print summary stats
print("\n=== Summary ===")
for method in ["qssa", "teacher", "student"]:
    overall = np.mean(rmse[method])
    in_zone = np.mean(rmse[method][mask_fail]) if mask_fail.any() else np.nan
    out_zone = np.mean(rmse[method][~mask_fail]) if (~mask_fail).any() else np.nan
    print(f"{method:8s}: overall={overall:.3e}, in_zone={in_zone:.3e}, out_zone={out_zone:.3e}")

print(f"\nQSSA failure zone: {mask_fail.sum()}/{len(mask_fail)} conditions")
