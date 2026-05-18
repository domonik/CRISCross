import pandas as pd

df = pd.read_csv("results/loso_results.csv")
df = df.sort_values("test_auprc", ascending=False)

print(df[["guide_id", "val_auprc", "test_auprc", "n_test_samples"]].to_string(index=False))
print(f"\nMean test AUPRC : {df['test_auprc'].mean():.4f}")
print(f"Std  test AUPRC : {df['test_auprc'].std():.4f}")
print(f"Best guide : {df.iloc[0]['guide_id']}  ({df.iloc[0]['test_auprc']:.4f})")
print(f"Worst guide: {df.iloc[-1]['guide_id']}  ({df.iloc[-1]['test_auprc']:.4f})")
