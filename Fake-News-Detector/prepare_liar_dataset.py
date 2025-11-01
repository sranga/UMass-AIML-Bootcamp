import pandas as pd

# Column names for LIAR dataset
columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job',
           'state', 'party', 'barely_true_count', 'false_count',
           'half_true_count', 'mostly_true_count', 'pants_fire_count', 'context']

# Load all three splits
train = pd.read_csv('data/train.tsv', sep='\t', header=None, names=columns)
valid = pd.read_csv('data/valid.tsv', sep='\t', header=None, names=columns)
test = pd.read_csv('data/test.tsv', sep='\t', header=None, names=columns)

print(f"Train size: {len(train)}")
print(f"Valid size: {len(valid)}")
print(f"Test size: {len(test)}")

# Combine all three datasets
df_combined = pd.concat([train, valid, test], ignore_index=True)
print(f"Combined size: {len(df_combined)}")

# Create binary labels
df_combined['binary_label'] = df_combined['label'].map({
    'pants-fire': 1,
    'false': 1,
    'barely-true': 1,
    'half-true': 0,
    'mostly-true': 0,
    'true': 0
})

# Check label distribution
print(f"\nOriginal label distribution:")
print(df_combined['label'].value_counts())
print(f"\nBinary label distribution:")
print(df_combined['binary_label'].value_counts())

# Clean and prepare for pipeline
df_clean = df_combined[['statement', 'binary_label']].dropna()
df_clean.columns = ['text', 'label']

# Remove very short statements
df_clean = df_clean[df_clean['text'].str.len() > 10]

# Shuffle
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset size: {len(df_clean)}")
print(f"Class balance:")
print(df_clean['label'].value_counts(normalize=True))

# Save combined dataset
df_clean.to_csv('data/liar_binary_combined.csv', index=False)

print("\n✅ Dataset saved as 'liar_binary_combined.csv'")
print("\nSample fake statement:")
print(df_clean[df_clean['label']==1]['text'].iloc[0])
print("\nSample real statement:")
print(df_clean[df_clean['label']==0]['text'].iloc[0])