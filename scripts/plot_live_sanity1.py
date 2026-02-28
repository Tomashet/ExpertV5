import argparse
import os
import time

import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--window", type=int, default=3000)
    parser.add_argument("--pause", type=float, default=0.8)

    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join("runs", args.run_dir, "train_monitor.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    plt.ion()

    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    while True:

        try:
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                time.sleep(args.pause)
                continue

            df = df.tail(args.window)

            t = df["timestep"].to_numpy()

            ctx = df.get("ctx_id", pd.Series([-1]*len(df))).to_numpy()

            viol = df.get("violation", pd.Series([0]*len(df))).astype(float).to_numpy()

            adj_risk = df.get("adj_risk", pd.Series([0]*len(df))).astype(float).to_numpy()

            # rolling violation rate
            viol_roll = pd.Series(viol).rolling(200, min_periods=1).mean().to_numpy()

            # clear plots
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # Panel 1: context
            ax1.plot(t, ctx)
            ax1.set_ylabel("ctx_id")
            ax1.set_title(os.path.basename(os.path.dirname(csv_path)))

            # Panel 2: violation rate
            ax2.plot(t, viol_roll)
            ax2.set_ylabel("violation rate")

            # Panel 3: adjustment-speed risk
            ax3.plot(t, adj_risk)
            ax3.set_ylabel("adj_risk")
            ax3.set_xlabel("timestep")

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(args.pause)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()