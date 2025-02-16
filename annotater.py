import logging
import pandas as pd
from transformers import pipeline
import time

# Configure logging to capture progress and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('annotater.log'),
        logging.StreamHandler()
    ]
)

# Candidate labels for annotation
CANDIDATE_LABELS = [
    "Deep Learning",
    "Computer Vision",
    "Reinforcement Learning",
    "NLP",
    "Optimization"
]

# File paths
INPUT_CSV = "papers.csv"
OUTPUT_CSV = "annotated_papers.csv"

def load_metadata(csv_file):
    """Load metadata from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_file}: {e}")
        raise

def initialize_classifier():
    """Initialize the zero-shot classification pipeline from Hugging Face."""
    try:
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        logging.info("Classifier initialized successfully.")
        return clf
    except Exception as e:
        logging.error(f"Error initializing classifier: {e}")
        raise

def annotate_paper(row, classifier):
    """
    Annotate a paper by combining its title and abstract, then classifying the text.
    Returns the predicted label with the highest confidence.
    """
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    # Combine title and abstract as context for classification.
    text = f"{title}\n\n{abstract}"
    
    try:
        result = classifier(text, candidate_labels=CANDIDATE_LABELS)
        # The classifier output includes sorted 'labels' (highest score first)
        predicted_label = result["labels"][0]
        logging.info(f"Annotated paper '{title}' as '{predicted_label}'.")
        return predicted_label
    except Exception as e:
        logging.error(f"Error annotating paper '{title}': {e}")
        return "Annotation_Error"

def save_metadata(df, csv_file):
    """Save the updated DataFrame to a CSV file."""
    try:
        df.to_csv(csv_file, index=False, encoding="utf-8")
        logging.info(f"Annotations saved to {csv_file}")
    except Exception as e:
        logging.error(f"Error saving annotations to CSV: {e}")
        raise

def main():
    # Load metadata from the CSV file
    df = load_metadata(INPUT_CSV)
    
    # Initialize the classifier
    classifier = initialize_classifier()
    
    # Add a new column for annotations
    annotations = []
    
    # Process each paper
    total = len(df)
    logging.info(f"Starting annotation of {total} papers.")
    start_time = time.time()
    
    for idx, row in df.iterrows():
        label = annotate_paper(row, classifier)
        annotations.append(label)
    
    # Append the annotations to the DataFrame
    df["annotation"] = annotations
    save_metadata(df, OUTPUT_CSV)
    
    elapsed = time.time() - start_time
    logging.info(f"Completed annotation of {total} papers in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
