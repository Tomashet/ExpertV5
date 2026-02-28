# scripts/plot_live_sanity.py
# Live plot: ctx_id (top) + violation rate (bottom) from runs/<run_dir>/train_monitor.csv
import argparse, os, time
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to train_monitor.csv (or pass --run_dir)")
    ap.add_argument("--run_dir", default="", help="runs/<run_dir> containing train_monitor.csv")
    ap.add_argument("--window", type=int, default=2000, help="How many recent rows to plot")
    ap.add_argument("--pause", type=float, default=0.8, help="Seconds between refresh")
    args = ap.parse_args()

    csv_path = args.csv or os.path.join("runs", args.run_dir, "train_monitor.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    while True:
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                time.sleep(args.pause); continue
            df = df.tail(args.window)

            t = df["timestep"].to_numpy()
            ctx = df.get("ctx_id", pd.Series([-1]*len(df))).to_numpy()
            viol = df.get("violation", pd.Series([0]*len(df))).astype(float).to_numpy()
            # rolling violation rate
            roll = pd.Series(viol).rolling(200, min_periods=1).mean().to_numpy()

            ax1.cla(); ax2.cla()
            ax1.plot(t, ctx)
            ax1.set_ylabel("ctx_id")
            ax1.set_title(os.path.basename(os.path.dirname(csv_path)) + "  (live)")

            ax2.plot(t, roll)
            ax2.set_ylabel("violation rate (roll=200)")
            ax2.set_xlabel("timestep")

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(args.pause)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()