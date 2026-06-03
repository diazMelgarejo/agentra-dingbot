"""
Monte Carlo simulation on backtest trade results.
Reads data/backtest_trades.csv and runs N random reshuffles
to estimate distribution of final bankroll and max drawdown.

Usage:
    python monte_carlo.py --simulations 5000
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")


def run_monte_carlo(csv_path: str = "data/backtest_trades.csv",
                    n_sims: int = 5000,
                    bankroll: float = 100.0) -> None:
    df = pd.read_csv(csv_path)
    pnl_series = df["pnl_usdc"].values

    if len(pnl_series) == 0:
        print("No trades in CSV")
        return

    final_bankrolls = []
    max_drawdowns = []

    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled = rng.choice(pnl_series, size=len(pnl_series), replace=True)
        equity = bankroll + np.cumsum(shuffled)
        equity = np.insert(equity, 0, bankroll)
        final_bankrolls.append(equity[-1])

        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
        max_drawdowns.append(max_dd)

    fb = np.array(final_bankrolls)
    dd = np.array(max_drawdowns)

    print("
" + "="*60)
    print(f"MONTE CARLO ({n_sims} simulations, {len(pnl_series)} trades reshuffled)")
    print("="*60)
    print(f"Median final bankroll : ${np.median(fb):.2f}")
    print(f"5th percentile         : ${np.percentile(fb, 5):.2f}")
    print(f"95th percentile        : ${np.percentile(fb, 95):.2f}")
    print(f"% sims profitable      : {(fb > bankroll).mean()*100:.1f}%")
    print(f"Median max drawdown    : {np.median(dd):.1f}%")
    print(f"95th pct max drawdown  : {np.percentile(dd, 95):.1f}%")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(fb, bins=100, color="#00c8ff", edgecolor="none", alpha=0.8)
    ax1.axvline(bankroll, color="red", linestyle="--", label="Start")
    ax1.set_title("Final Bankroll Distribution")
    ax1.set_xlabel("USDC")
    ax1.legend()

    ax2.hist(dd, bins=100, color="#ff6b6b", edgecolor="none", alpha=0.8)
    ax2.set_title("Max Drawdown Distribution (%)")
    ax2.set_xlabel("%")

    plt.tight_layout()
    plt.savefig("data/monte_carlo.png", dpi=150)
    print("Saved: data/monte_carlo.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--bankroll",    type=float, default=100.0)
    args = parser.parse_args()
    run_monte_carlo(n_sims=args.simulations, bankroll=args.bankroll)
