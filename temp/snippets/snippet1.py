from sklearn.model_selection import train_test_split

# 1. Compress down to unique interactions (just like you did!)
interaction_df = df.groupby("interaction_id").first().reset_index()

# 2. First Split: Extract the Training set (70%) and a Temporary set (30%)
train_ids_df, temp_ids_df = train_test_split(
    interaction_df,                  # Pass the whole grouped dataframe
    test_size=0.30,                  # 30% goes to the temp set (Val + Test)
    stratify=interaction_df["object"],
    random_state=42
)

# 3. Second Split: Divide the Temporary set perfectly in half (15% Val / 15% Test)
val_ids_df, test_ids_df = train_test_split(
    temp_ids_df,
    test_size=0.50,                  # 50% of the 30% temp set = 15% of the total
    stratify=temp_ids_df["object"],  # Stratify again using the temp set's labels!
    random_state=42
)

# 4. Map the safely split IDs back to the original full-frame dataset
train_df = df[df["interaction_id"].isin(train_ids_df["interaction_id"])]
val_df = df[df["interaction_id"].isin(val_ids_df["interaction_id"])]
test_df = df[df["interaction_id"].isin(test_ids_df["interaction_id"])]

print(f"Train frames: {len(train_df)}")
print(f"Val frames:   {len(val_df)}")
print(f"Test frames:  {len(test_df)}")