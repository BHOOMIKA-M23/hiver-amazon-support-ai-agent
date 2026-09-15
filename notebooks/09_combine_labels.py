import pandas as pd

file1 = "data/intent_labeling_sample.csv"
file2 = "data/amazon_label_batch_2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

combined = pd.concat([df1, df2], ignore_index=True)

output_file = "data/amazon_golden_set.csv"
combined.to_csv(output_file, index=False)

print("Total examples:", len(combined))
print("Saved to:", output_file)