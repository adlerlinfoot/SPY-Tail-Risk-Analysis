# SPY-Tail-Risk-Analysis

Empirical analysis of SPY daily returns: Exploring the limits of static and rolling Sharpe ratios in leptokurtic return distributions.



OVERVIEW

This project evaluates how SPY’s empirical return distribution deviates normal-model assumptions and how traditional risk metrics such as static, rolling, and tail-adjusted Sharpe ratios fail to capture downside risk.

The pipeline computes empirical tail losses, compares them to a normal benchmark, and visualizes distributional deviations and regime‑dependent Sharpe dynamics using a rolling window.



KEY FINDINGS

• During 1% worst days, the normal model understates tail losses by ~59%(ES*1*% Empirical -5.00% vs. Normal -3.14%).

• Tail-adjusted Sharpe (-4.11) reveals far more downside risk than static Sharpe (0.05).

• Empirical skew is negative (-0.30), indicating left-tail bias.

• 3% tail exceedances occur 2.8x more often than normal predicts, consistent with high excess kurtosis (13.65).

These deviations highlight the pitfalls of the Sharpe ratio in heavy-tailed distributions where rare but severe losses dominate risk.



REPRODUCIBILITY

Run the full analysis from the project root: python scripts/tail\_analysis.py



REQUIREMENTS

Python 3.10+ with standard scientific libraries (NumPy, Pandas, Matplotlib, SciPy).



FULL REPORT

A concise summary of methods and results is available in: report/report.pdf

