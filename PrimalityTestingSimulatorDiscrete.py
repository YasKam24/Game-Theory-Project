import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sympy import isprime, randprime

# --- Constants ---
R = 2.0
S = 1.0

# Modified for 10x10 grid (100 cells)
DIGIT_RANGE = range(10, 101, 10)    # 10, 20, ..., 100 (10 steps)
LAMBDA_POWERS = np.arange(1, 11)    # 10^1, 10^2, ..., 10^10 (10 steps)

def measure_cost(n):
    """Measures CPU time for primality testing."""
    start = time.perf_counter()
    isprime(n)
    return time.perf_counter() - start

def run_simulation():
    results = []
    print(f"Starting 10x10 Simulation: R={R}, S={S}")
    
    for p in LAMBDA_POWERS:
        lam = 10**p
        print(f"  Testing Lambda = 10^{p}...")
        for digits in DIGIT_RANGE:
            # Sample numbers to get an average cost
            # Note: 100-digit primality tests are fast, but many samples improve stability
            costs = [measure_cost(randprime(10**(digits-1), 10**digits)) for _ in range(5)]
            avg_cost = np.mean(costs)
            
            # Logic: Compute if Utility(Compute) > Utility(Safe)
            # (R - lam * cost) > S
            u_compute = R - (lam * avg_cost)
            strategy = 1 if u_compute > S else 0
            
            results.append({
                'log_lambda': f"10^{p}",
                'digits': digits,
                'strategy': strategy,
                'p_val': p # Keep for sorting
            })
            
    return pd.DataFrame(results)

def plot_strategies(df):
    plt.figure(figsize=(12, 8))
    
    # Ensure the Y-axis follows numeric order of powers
    df = df.sort_values(by=['p_val', 'digits'])
    pivot = df.pivot(index="log_lambda", columns="digits", values="strategy")
    # Re-indexing to ensure 10^1 is at the bottom and 10^10 is at the top
    pivot = pivot.reindex([f"10^{p}" for p in reversed(LAMBDA_POWERS)])
    
    # Heatmap: Green = Compute (1), Red = Safe (0)
    sns.heatmap(pivot, cmap="RdYlGn", cbar=False, linewidths=1, linecolor='white', annot=False)
    
    plt.title(f"10x10 Primality Decision Game (R={R}, S={S})\nGreen = Compute | Red = Play Safe", fontsize=16)
    plt.xlabel("Number Size (Digits)", fontsize=12)
    plt.ylabel("Cost Weight (λ)", fontsize=12)
    
    output = "strategy_map_10x10.png"
    plt.savefig(output, dpi=300)
    print(f"\n✓ 100-cell strategy map saved as: {output}")
    plt.show()

if __name__ == "__main__":
    data = run_simulation()
    plot_strategies(data)