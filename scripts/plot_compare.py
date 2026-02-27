import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_curve(run_dir: str, metric: str, step_col: str = "timestep"):
    csv_path = os.path.join("runs", run_dir, "train_monitor.csv")
    df = pd.read_csv(csv_path)
    if step_col not in df.columns:
        raise ValueError(f"{run_dir}: missing '{step_col}'")
    if metric not in df.columns:
        raise ValueError(f"{run_dir}: missing '{metric}'")
    return df[step_col].to_numpy(), df[metric].to_numpy()

def resample_to_grid(x, y, grid):
    # y is piecewise-constant between recorded x; use interpolation
    # (works well for step-indexed logs)
    return np.interp(grid, x, y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True,
                    help="Pairs: name=pattern. Example: baseline=B_dqn_seed* adj=F_dqn_seed*")
    ap.add_argument("--metric", default="violation", help="e.g., violation, adj_risk, shield_used")
    ap.add_argument("--smooth", type=int, default=2000, help="rolling window (in samples on resampled grid)")
    ap.add_argument("--grid_points", type=int, default=2000, help="resample points")
    args = ap.parse_args()

    groups = []
    for item in args.groups:
        if "=" not in item:
            raise ValueError("Group must be name=pattern, e.g., baseline=B_dqn_seed*")
        name, pattern = item.split("=", 1)
        run_dirs = [os.path.basename(p) for p in glob.glob(os.path.join("runs", pattern))]
        if not run_dirs:
            raise ValueError(f"No runs matched pattern: {pattern}")
        groups.append((name, run_dirs))

    # Build a common timestep grid from all runs
    all_max = []
    all_min = []
    for _, rds in groups:
        for rd in rds:
            x, y = load_curve(rd, args.metric)
            all_min.append(x.min())
            all_max.append(x.max())
    x0 = float(max(all_min))
    x1 = float(min(all_max))
    if x1 <= x0:
        raise ValueError("Runs do not overlap in timesteps enough to compare.")

    grid = np.linspace(x0, x1, args.grid_points)

    plt.figure()

    for name, rds in groups:
        Ys = []
        for rd in rds:
            x, y = load_curve(rd, args.metric)
            y = y.astype(float)
            y_grid = resample_to_grid(x, y, grid)
            Ys.append(y_grid)
        Y = np.vstack(Ys)
        mean = Y.mean(axis=0)
        se = Y.std(axis=0) / np.sqrt(Y.shape[0])

        # smoothing (simple moving average on the grid)
        if args.smooth > 1:
            w = min(args.smooth, len(mean))
            kernel = np.ones(w) / w
            mean_s = np.convolve(mean, kernel, mode="same")
            se_s = np.convolve(se, kernel, mode="same")
        else:
            mean_s, se_s = mean, se

        plt.plot(grid, mean_s, label=f"{name} (n={len(rds)})")
        plt.fill_between(grid, mean_s - se_s, mean_s + se_s, alpha=0.2)

    plt.xlabel("timestep")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric}: mean ± SE across seeds")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()