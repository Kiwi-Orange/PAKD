#!/usr/bin/env python3
"""Quick test-set generalization comparison: Full vs No hidden distillation."""

import numpy as np
import torch
from pathlib import Path
import make_nature_figure as mn
from models import ResidualMLP

ROOT = Path(__file__).resolve().parent
ABLATION_DIR = ROOT / "results" / "ablation"

def evaluate_student_on_conditions(student_path, conditions):
    """Evaluate a saved student model on held-out conditions."""
    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=False)
    
    # Reconstruct student model
    student = ResidualMLP(
        input_size=5, output_size=4,
        hidden_dim=64, num_blocks=3, dropout=0.0,
    )
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    
    X_scaler = ckpt["X_scaler"]
    y_scaler = ckpt.get("y_scaler", None)
    
    # Teacher bundle (model, x_scaler, y_scaler, checkpoint)
    teacher_bundle = mn.load_checkpoint_model(mn.TEACHER_MODEL, is_student=False)
    student_bundle = (student, X_scaler, y_scaler, ckpt)
    
    trajectories = mn.compute_all_trajectories(teacher_bundle, student_bundle, conditions)
    truth = trajectories["analytical"]
    student_traj = trajectories["student"]
    teacher_traj = trajectories["teacher"]
    
    # Per-condition RMSE
    student_rmse = np.sqrt(np.mean((student_traj - truth) ** 2, axis=(1, 2)))
    teacher_rmse = np.sqrt(np.mean((teacher_traj - truth) ** 2, axis=(1, 2)))
    
    return {
        "student_mean": student_rmse.mean(),
        "student_std": student_rmse.std(),
        "teacher_mean": teacher_rmse.mean(),
        "teacher_std": teacher_rmse.std(),
        "student_per_condition": student_rmse,
        "teacher_per_condition": teacher_rmse,
    }


def main():
    print("=" * 60)
    print("Test-Set Generalization: Full vs No hidden")
    print("=" * 60)
    
    # Generate 50 held-out test conditions
    conditions = mn.generate_test_conditions(50)
    print(f"Test conditions: {conditions.shape[0]} initial conditions")
    
    variants = [
        ("Full", ABLATION_DIR / "student_full.pt"),
        ("No hidden", ABLATION_DIR / "student_no_hidden.pt"),
    ]
    
    results = {}
    for name, path in variants:
        print(f"\nEvaluating {name}...")
        r = evaluate_student_on_conditions(path, conditions)
        results[name] = r
        print(f"  Student RMSE: {r['student_mean']:.6f} ± {r['student_std']:.6f}")
        print(f"  Teacher RMSE: {r['teacher_mean']:.6f} ± {r['teacher_std']:.6f}")
    
    # Comparison
    full = results["Full"]
    no_hidden = results["No hidden"]
    
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Full PAKD     : {full['student_mean']:.6f}")
    print(f"No hidden     : {no_hidden['student_mean']:.6f}")
    
    diff = full['student_mean'] - no_hidden['student_mean']
    pct = (diff / no_hidden['student_mean']) * 100
    
    if diff > 0:
        print(f"No hidden is BETTER by {abs(diff):.6f} ({abs(pct):.2f}%)")
    elif diff < 0:
        print(f"Full PAKD is BETTER by {abs(diff):.6f} ({abs(pct):.2f}%)")
    else:
        print("No difference")
    
    # Per-condition comparison
    per_condition_diff = full['student_per_condition'] - no_hidden['student_per_condition']
    n_full_better = np.sum(per_condition_diff < 0)
    n_no_hidden_better = np.sum(per_condition_diff > 0)
    print(f"\nPer-condition wins:")
    print(f"  Full PAKD better: {n_full_better} / 50")
    print(f"  No hidden better: {n_no_hidden_better} / 50")


if __name__ == "__main__":
    main()
