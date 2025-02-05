import pandas as pd
import random

def create_similarity_dataset(input_file, output_file, max_samples=5000):
    """
    Create a dataset of comment pairs with similarity labels.
    :param input_file: Path to reddit_data.csv file.
    :param output_file: Path to save similarity_dataset.csv file.
    :param max_samples: Maximum number of pairs to generate.
    """
    try:
        # Load Reddit data
        df = pd.read_csv(input_file)

        # Ensure required columns exist
        if 'comment_body' not in df.columns or 'subreddit' not in df.columns:
            raise ValueError("The input file must have 'comment_body' and 'subreddit' columns.")

        # Extract comments and subreddits
        comments = df['comment_body'].dropna().tolist()
        subreddits = df['subreddit'].tolist()

        # Create positive pairs (same subreddit)
        positive_pairs = []
        for i in range(len(comments) - 1):
            if subreddits[i] == subreddits[i + 1]:
                positive_pairs.append((comments[i], comments[i + 1], 1))

        # Create negative pairs (different subreddits)
        negative_pairs = []
        for _ in range(len(positive_pairs)):
            c1, c2 = random.sample(comments, 2)
            negative_pairs.append((c1, c2, 0))

        # Combine and shuffle pairs
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        # Limit the number of samples
        all_pairs = all_pairs[:max_samples]

        # Save to CSV
        similarity_df = pd.DataFrame(all_pairs, columns=["Comment1", "Comment2", "Label"])
        similarity_df.to_csv(output_file, index=False)
        print(f"Similarity dataset saved to {output_file}.")
    except Exception as e:
        print(f"Error creating similarity dataset: {e}")

# Generate similarity dataset
input_file = "data/reddit_data.csv"
output_file = "data/similarity_dataset.csv"
create_similarity_dataset(input_file, output_file)
