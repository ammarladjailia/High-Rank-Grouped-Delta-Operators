import numpy as np
import pandas as pd
from scipy import stats

def holm_bonferroni(p_values):
    """
    Applies Holm-Bonferroni correction to a list of p-values.
    Returns array of adjusted p-values in original order.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        # multiplier is (n - i)
        val = sorted_p[i] * (n - i)
        adjusted_sorted[i] = val
        
    # Enforce monotonicity
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])
        
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    
    # Restore original order
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_indices] = adjusted_sorted
    return adjusted_p

def analyze_comparisons():
    # Seeds: [42, 123, 456, 789, 2026]
    hrg_beta02 = np.array([93.74, 93.60, 93.70, 93.49, 93.73])
    hrg_beta01 = np.array([93.58, 93.50, 93.63, 93.66, 93.52])
    param_matched = np.array([93.49, 93.35, 93.65, 93.63, 93.32])
    senet = np.array([93.00, 92.87, 92.83, 93.15, 92.65]) # wait, let's check SE-Net and CA-Net exact 5 seeds
    canet = np.array([93.41, 93.23, 93.26, 93.54, 93.55])
    vanilla = np.array([92.75, 92.51, 92.36, 92.40, 92.34])
    beta03 = np.array([93.68, 94.10, 93.73, 93.62, 93.95])
    rank1_g4 = np.array([93.20, 93.42, 93.73, 93.45, 93.67])
    
    comparisons = [
        ("HRG-Delta [0,2] vs [0,1]", hrg_beta02, hrg_beta01),
        ("HRG-Delta [0,2] vs Param-matched", hrg_beta02, param_matched),
        ("HRG-Delta [0,2] vs SE-Net", hrg_beta02, senet),
        ("HRG-Delta [0,2] vs CA-Net", hrg_beta02, canet),
        ("HRG-Delta [0,2] vs Vanilla ResNet", hrg_beta02, vanilla),
        ("HRG-Delta [0,2] vs Rank-1 Grouped (G=4,r=1)", hrg_beta02, rank1_g4),
        ("HRG-Delta [0,2] vs Beta_max=3", hrg_beta02, beta03),
    ]
    
    records = []
    raw_p_paired = []
    raw_p_unpaired = []
    
    for name, grp1, grp2 in comparisons:
        diff = grp1 - grp2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        # Paired t-test
        t_paired, p_paired = stats.ttest_rel(grp1, grp2)
        ci_paired = stats.t.interval(0.95, df=len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
        d_z = mean_diff / std_diff if std_diff > 0 else 0.0
        
        # Unpaired Welch's t-test
        t_unpaired, p_unpaired = stats.ttest_ind(grp1, grp2, equal_var=False)
        pooled_std = np.sqrt((np.var(grp1, ddof=1) + np.var(grp2, ddof=1)) / 2)
        d_cohen = (np.mean(grp1) - np.mean(grp2)) / pooled_std
        
        raw_p_paired.append(p_paired)
        raw_p_unpaired.append(p_unpaired)
        
        records.append({
            "Comparison": name,
            "Mean_1": np.mean(grp1),
            "Std_1": np.std(grp1, ddof=1),
            "Mean_2": np.mean(grp2),
            "Std_2": np.std(grp2, ddof=1),
            "Mean_Diff": mean_diff,
            "CI95_Low": ci_paired[0],
            "CI95_High": ci_paired[1],
            "t_paired": t_paired,
            "df": len(diff) - 1,
            "p_paired_raw": p_paired,
            "d_z": d_z,
            "t_unpaired": t_unpaired,
            "p_unpaired_raw": p_unpaired,
            "d_cohen": d_cohen
        })
        
    p_paired_holm = holm_bonferroni(raw_p_paired)
    p_unpaired_holm = holm_bonferroni(raw_p_unpaired)
    
    for i, r in enumerate(records):
        r["p_paired_Holm"] = p_paired_holm[i]
        r["p_unpaired_Holm"] = p_unpaired_holm[i]
        
    df = pd.DataFrame(records)
    df.to_csv("/mnt/win/Ammar/Delta/new_Implementation/statistical_tests_cifar10_accuracy.csv", index=False)
    
    print("=== SUMMARY OF STATISTICAL TESTS ON TEST ACCURACY ===")
    print(df[["Comparison", "Mean_Diff", "CI95_Low", "CI95_High", "t_paired", "p_paired_raw", "p_paired_Holm"]].to_string())

if __name__ == "__main__":
    analyze_comparisons()
