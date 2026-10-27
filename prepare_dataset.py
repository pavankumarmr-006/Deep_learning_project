
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Create sample support tickets
categories = {
    "billing": [
        "I was charged twice for my order",
        "My invoice amount is incorrect",
        "I need a refund for my payment",
        "My payment failed during checkout",
    ],
    "account": [
        "I cannot log in to my account",
        "Please help me reset my password",
        "I want to update my account details",
        "My account has been locked",
    ],
    "technical": [
        "The application crashes when I open it",
        "The website is loading very slowly",
        "I get an error when using the app",
        "The latest update broke a feature",
    ],
    "delivery": [
        "My package has not arrived",
        "My order is delayed",
        "The tracking page is not updating",
        "I received the wrong item",
    ],
    "returns": [
        "I want to return my purchase",
        "How can I exchange this item",
        "My returned item has not been refunded",
        "The product arrived damaged and needs returning",
    ],
}

priorities = ["low", "medium", "high"]

rows = []

for category, messages in categories.items():
    for priority in priorities:
        for i in range(40):
            message = messages[i % len(messages)]

            # Add a small variation to create distinct examples
            text = f"{message}. Support reference {category}-{priority}-{i + 1}."

            rows.append({
                "text": text,
                "category": category,
                "priority": priority,
            })

df = pd.DataFrame(rows)

# 2. Clean the text
df["text"] = df["text"].str.lower().str.strip()
df["category"] = df["category"].str.lower().str.strip()
df["priority"] = df["priority"].str.lower().str.strip()

# Remove rows with missing required values
df = df.dropna(subset=["text", "category", "priority"])

# Remove duplicate rows
df = df.drop_duplicates(subset=["text", "category", "priority"])

# Save the cleaned dataset
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
df.to_csv(data_dir / "tickets.csv", index=False)

# 3. Print dataset summary
print("\nDATASET SUMMARY")
print("----------------")
print("Total tickets:", len(df))

print("\nCategory counts:")
print(df["category"].value_counts().sort_index())

print("\nPriority counts:")
print(df["priority"].value_counts().sort_index())

# 4. Split into 70% training and 30% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["category"] + "_" + df["priority"],
)

# Split the temporary data equally into validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["category"] + "_" + temp_df["priority"],
)

# 5. Save the three datasets
train_df.to_csv(data_dir / "train.csv", index=False)
val_df.to_csv(data_dir / "validation.csv", index=False)
test_df.to_csv(data_dir / "test.csv", index=False)

print("\nSPLIT SUMMARY")
print("--------------")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# Show label distributions in each split
for name, split in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df),
]:
    print(f"\n{name} category counts:")
    print(split["category"].value_counts().sort_index())

    print(f"\n{name} priority counts:")
    print(split["priority"].value_counts().sort_index())

print("\nDataset preparation completed!")
