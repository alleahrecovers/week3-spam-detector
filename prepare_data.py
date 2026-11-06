
import pandas as pd
import os

# Create the data folder if it does not exist
os.makedirs("data", exist_ok=True)

# Create 16 sample emails
emails = [
    # Spam emails
    ("CONGRATULATIONS! You won $5000! Claim your prize NOW!!!", 1),
    ("URGENT! Claim your FREE gift card today!", 1),
    ("Make MONEY fast with this amazing opportunity!", 1),
    ("You have WON a lottery prize! Click here to claim!", 1),
    ("Limited time offer! Buy now and SAVE BIG!!!", 1),
    ("FREE cash waiting for you. Act immediately!", 1),
    ("Earn $$$ working from home. Sign up TODAY!", 1),
    ("Exclusive deal! Get rich quickly with no effort!", 1),

    # Legitimate emails
    ("Hi, can we schedule a meeting for tomorrow?", 0),
    ("Please review the attached project report.", 0),
    ("Thank you for your help with the assignment.", 0),
    ("The team meeting has been moved to Friday.", 0),
    ("I will send you the notes after class.", 0),
    ("Your appointment is confirmed for Monday.", 0),
    ("Could you please call me when you are available?", 0),
    ("Here is the updated schedule for next week.", 0)
]

df = pd.DataFrame(emails, columns=["email", "label"])

# Save the dataset
df.to_csv("data/emails.csv", index=False)

print("Dataset created successfully!")
print(df["label"].value_counts())
