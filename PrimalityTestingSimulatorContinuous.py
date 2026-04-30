import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sympy import isprime, randprime

# --- User-Provided Data ---
# Cost Weights (Lambda)
LAMBDA_VALS = np.array([
    10.00, 13.28, 17.63, 23.42, 31.10, 41.30, 54.85, 72.85, 96.75, 128.49, 170.64, 
    226.62, 301.00, 399.73, 530.88, 705.07, 936.40, 1243.63, 1651.66, 2193.57, 
    2913.27, 3869.12, 5138.56, 6824.51, 9063.59, 12037.27, 15986.60, 21231.63, 
    28197.54, 37448.91, 49735.19, 66052.81, 87723.95, 116505.72, 154730.01, 
    205494.38, 272914.13, 362455.51, 481374.07, 639309.11, 849052.88, 1127617.91, 
    1497575.46, 1988913.80, 2641452.88, 3508088.16, 4659017.52, 6187585.57, 
    8217664.38, 10000000.00
])

# Number Sizes (Digits) - skipping 0.00 to avoid math errors (min 1 digit)
DIGIT_SAMPLES = np.array([
    2.04, 4.08, 6.12, 8.16, 10.20, 12.24, 14.29, 16.33, 18.37, 20.41, 22.45, 24.49, 
    26.53, 28.57, 30.61, 32.65, 34.69, 36.73, 38.78, 40.82, 42.86, 44.90, 46.94, 
    48.98, 51.02, 53.06, 55.10, 57.14, 59.18, 61.22, 63.27, 65.31, 67.35, 69.39, 
    71.43, 73.47, 75.51, 77.55, 79.59, 81.63, 83.67, 85.71, 87.76, 89.80, 91.84, 
    93.88, 95.92, 97.96, 100.00
])

R = 2.0
S = 1.0

def measure_cost(d):
    """Measures CPU time for a random prime of digit size d."""
    # Ensure d is at least 1 and an integer
    d_int = max(1, int(round(d)))
    n = randprime(10**(d_int-1), 10**d_int)
    start = time.perf_counter()
    isprime(n)
    return time.perf_counter() - start

def run_simulation():
    results = []
    print(f"Running Simulation with custom grid...")
    
    # Pre-calculate costs for digit samples
    digit_costs = {d: np.mean([measure_cost(d) for _ in range(5)]) for d in DIGIT_SAMPLES}

    for lam in LAMBDA_VALS:
        for d in DIGIT_SAMPLES:
            avg_cost = digit_costs[d]
            u_diff = (R - lam * avg_cost) - S
            strategy = 1 if u_diff > 0 else 0
            
            results.append({
                'lambda': lam,
                'log_lambda': np.log10(lam),
                'digits': d,
                'strategy': strategy,
                'u_diff': u_diff
            })
    return pd.DataFrame(results)

def plot_results(df):
    # Pivot for heatmap and boundary
    pivot_strat = df.pivot(index="log_lambda", columns="digits", values="strategy")
    pivot_diff = df.pivot(index="log_lambda", columns="digits", values="u_diff")
    
    plt.figure(figsize=(14, 9))
    
    # Plot heatmap (Green=Compute, Red=Safe)
    ax = sns.heatmap(pivot_strat, cmap="RdYlGn", cbar=False, alpha=0.8)
    
    # Boundary Line (Contour)
    X = np.arange(len(pivot_strat.columns)) + 0.5
    Y = np.arange(len(pivot_strat.index)) + 0.5
    plt.contour(X, Y, pivot_diff.values, levels=[0], colors='black', linewidths=4)
    
    # Labeling
    plt.title(f"Strategic Map: Compute vs Safe\nBoundary Highlighted (R={R}, S={S})", fontsize=16)
    plt.xlabel("Number Size (Digits)", fontsize=12)
    plt.ylabel("Cost Weight λ (Scale)", fontsize=12)
    
    # Format Y-axis to show readable lambda values
    y_indices = np.linspace(0, len(LAMBDA_VALS)-1, 10).astype(int)
    plt.yticks(y_indices + 0.5, [f"{LAMBDA_VALS[i]:.2e}" for i in y_indices], rotation=0)
    
    # Format X-axis
    x_indices = np.linspace(0, len(DIGIT_SAMPLES)-1, 10).astype(int)
    plt.xticks(x_indices + 0.5, [f"{DIGIT_SAMPLES[i]:.1f}" for i in x_indices])

    plt.tight_layout()
    plt.savefig("custom_strategy_map.png", dpi=300)
    print("\n✓ Simulation finished. File: custom_strategy_map.png")
    plt.show()

if __name__ == "__main__":
    df = run_simulation()
    plot_results(df)