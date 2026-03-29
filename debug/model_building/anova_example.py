import os
import pandas as pd
import numpy as np
from scipy import stats


def anova_manual(groups):
    lengths = [len(g) for g in groups]
    N = sum(lengths)
    group_means = [np.mean(g) if len(g) > 0 else np.nan for g in groups]
    overall_mean = sum(m * n for m, n in zip(group_means, lengths)) / N
    SSB = sum(n * (m - overall_mean) ** 2 for n, m in zip(lengths, group_means))
    SSW = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    k = len(groups)
    dfb = k - 1
    dfw = N - k
    MSB = SSB / dfb if dfb > 0 else np.nan
    MSW = SSW / dfw if dfw > 0 else np.nan
    F = MSB / MSW if MSW > 0 else np.nan
    p = stats.f.sf(F, dfb, dfw) if not np.isnan(F) else np.nan
    return {
        'N': N,
        'k': k,
        'SSB': SSB,
        'SSW': SSW,
        'MSB': MSB,
        'MSW': MSW,
        'F': F,
        'p_value': p,
    }


def main():
    df = pd.read_csv('diabetes.csv')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')

    results = []
    for col in numeric_cols:
        g0 = df.loc[df['Outcome'] == 0, col].dropna().values
        g1 = df.loc[df['Outcome'] == 1, col].dropna().values
        res = anova_manual([g0, g1])
        res['feature'] = col
        # Supplemental: two-sample t-test for comparison (equivalent when k=2)
        try:
            t_stat, t_p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy='omit')
        except Exception:
            t_stat, t_p = np.nan, np.nan
        res['t_stat'] = t_stat
        res['t_p'] = t_p
        results.append(res)

    out_df = pd.DataFrame(results)
    out_dir = os.path.join('debug', 'model_building')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'anova_results.csv')
    out_df = out_df[['feature', 'N', 'k', 'SSB', 'SSW', 'MSB', 'MSW', 'F', 'p_value', 't_stat', 't_p']]
    out_df.to_csv(out_csv, index=False)
    print(f"ANOVA results saved to: {out_csv}")
    print(out_df.to_string(index=False, float_format='%.6f'))


if __name__ == '__main__':
    main()
