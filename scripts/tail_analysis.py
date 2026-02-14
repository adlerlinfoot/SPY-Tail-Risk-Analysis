# scripts/tail_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from pathlib import Path
import statsmodels.api as sm
import tempfile
import shutil

# Rolling and tail sharpe definitions
def rolling_sharpe(returns, window=252):
    """
    Compute rolling Sharpe ratio using a simple rolling mean/std.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = roll_mean / roll_std
    return sharpe


def rolling_tail_sharpe(returns, window=252, alpha=0.05):
    """
    Compute rolling tail-adjusted Sharpe ratio using ES in the denominator.
    """
    tail_sharpes = []
    index = []

    for i in range(window, len(returns)):
        window_data = returns[i-window:i]
        mu = window_data.mean()
        es = window_data[window_data <= window_data.quantile(alpha)].mean()

        if es == 0 or np.isnan(es):
            tail_sharpes.append(np.nan)
        else:
            tail_sharpes.append(mu / abs(es))

        index.append(returns.index[i])

    return pd.Series(tail_sharpes, index=index)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SPY.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Load and prepare
df = pd.read_csv(DATA)
date_col = [c for c in df.columns if any(k in c.lower() for k in ("date","time","timestamp"))][0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)
price_col = "close" if "close" in df.columns else df.select_dtypes(include=["number"]).columns[0]
prices = df[price_col].astype(float)
r = prices.pct_change().dropna()

# Dynamic sharpe calculations
roll_sharpe = rolling_sharpe(r, window=252)
roll_tail_sharpe = rolling_tail_sharpe(r, window=252, alpha=0.05)

# Basic stats
mu = r.mean()
sigma = r.std(ddof=1)
n = len(r)

# Save basic stats
with open(ROOT / "spy_stats.txt", "w") as f:
    f.write(f"Date range: {r.index.min().date()} to {r.index.max().date()}\n")
    f.write(f"Observations: {n}\n")
    f.write(f"Mean: {mu:.6f}\nStd dev: {sigma:.6f}\nSkewness: {r.skew():.6f}\nExcess kurtosis: {r.kurtosis():.6f}\n")

# 1) Histogram + normal overlay
plt.figure(figsize=(8,5))
bins = 80
plt.hist(r, bins=bins, density=True, alpha=0.6, color='C0', label='Empirical')
x = np.linspace(r.min(), r.max(), 1000)
pdf = st.norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x, pdf, 'r-', lw=2, label='Normal PDF (empirical μ,σ)')
plt.xlabel('Daily return')
plt.ylabel('Density')
plt.title('SPY daily returns: histogram + normal overlay')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "hist_normal_overlay.png", dpi=200)
plt.close()

# 2) QQ plot vs normal
plt.figure(figsize=(6,6))
sm.qqplot(r, line='45', dist=st.norm, fit=True)
plt.title('QQ plot: empirical vs normal')
plt.tight_layout()
plt.savefig(FIGS / "qq_plot.png", dpi=200)
plt.close()

# 3) Tail exceedance plot (absolute returns)
thresholds = np.linspace(0.001, 0.10, 100)
empirical_abs = np.array([(r.abs() > t).mean() for t in thresholds])
# two-sided normal exceedance using fitted mu and sigma
normal_abs = np.array([2*(1 - st.norm.cdf(t, loc=0, scale=sigma)) for t in thresholds])
# Note: normal_abs uses scale=sigma and center 0 because thresholds are absolute; this matches plotting convention
plt.figure(figsize=(8,5))
plt.semilogy(thresholds*100, empirical_abs, label='Empirical P(|r| > threshold)')
plt.semilogy(thresholds*100, normal_abs, '--', label='Normal model P(|r| > threshold)')
plt.xlabel('Threshold (% absolute daily return)')
plt.ylabel('Probability (log scale)')
plt.title('Tail exceedance: empirical vs normal')
plt.legend()
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(FIGS / "tail_exceedance.png", dpi=200)
plt.close()

# 4) Dynamic sharpe figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(roll_sharpe, label='Rolling Sharpe (252d)', color='blue')
plt.title('Rolling Sharpe Ratio (252-Day Window)')
plt.ylabel('Sharpe')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(roll_tail_sharpe, label='Rolling Tail-Sharpe (ES-Based)', color='red')
plt.title('Rolling Tail-Adjusted Sharpe Ratio (ES-Based)')
plt.ylabel('Tail-Sharpe')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(FIGS / "dynamic_sharpe.png", dpi=300)
plt.close()

# 5) Numeric tail comparisons for chosen thresholds (left tail losses)
chosen = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
rows = []
for t in chosen:
    emp = (r <= -t).mean()  # empirical left-tail
    norm_pred = st.norm.cdf(-t, loc=mu, scale=sigma)  # CORRECT left-tail probability
    ratio = (emp / norm_pred) if norm_pred > 0 else np.nan
    rows.append((t, emp, norm_pred, ratio))

# write atomically to avoid lock issues
tmp = ROOT / "tail_comparison.tmp"
with open(tmp, "w") as f:
    f.write("threshold,empirical_left_tail,normal_predicted_left_tail,ratio(emp/norm)\n")
    for t, emp, norm_pred, ratio in rows:
        f.write(f"{t},{emp:.6e},{norm_pred:.6e},{ratio:.3f}\n")
shutil.move(str(tmp), str(ROOT / "tail_comparison.txt"))

# 6) Expected Shortfall (ES) at alpha levels (empirical vs normal)
alphas = [0.01, 0.05]
tmp2 = ROOT / "es_comparison.tmp"
with open(tmp2, "w") as f:
    f.write("alpha,ES_empirical,ES_normal\n")
    for a in alphas:
        var_emp = np.quantile(r, a)
        es_emp = r[r <= var_emp].mean()
        # normal ES for left tail at level a for N(mu, sigma)
        z0 = st.norm.ppf(a)
        # ES_normal = mu + sigma * (pdf(z0) / a) * (-1)  (standard formula for left tail)
        es_norm = mu - sigma * (st.norm.pdf(z0) / a)
        f.write(f"{a},{es_emp:.6f},{es_norm:.6f}\n")
shutil.move(str(tmp2), str(ROOT / "es_comparison.txt"))

# 7) Sharpe and a simple tail-adjusted Sharpe (penalize by ES magnitude)
sharpe = mu / sigma
var1 = np.quantile(r, 0.01)
es1 = r[r <= var1].mean()
tail_adj_sharpe = (mu - abs(es1)) / sigma

tmp3 = ROOT / "sharpe_comparison.tmp"
with open(tmp3, "w") as f:
    f.write(f"Sharpe (mean/std): {sharpe:.6f}\n")
    f.write(f"ES(1%): {es1:.6f}\n")
    f.write(f"Tail-adjusted Sharpe (mean - |ES(1%)|)/std: {tail_adj_sharpe:.6f}\n")
shutil.move(str(tmp3), str(ROOT / "sharpe_comparison.txt"))

print("Done. Figures and numeric outputs saved to figures/ and root files.")