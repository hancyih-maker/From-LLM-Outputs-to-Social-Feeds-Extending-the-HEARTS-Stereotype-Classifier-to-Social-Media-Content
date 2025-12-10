import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch

# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {df.shape[0]} observations.")
    return df

# Class to handle sentiment analysis
class SentimentClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        if device >= 0:
            model.to(f'cuda:{device}')
        self.sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, device=device)

    def compute_sentiment(self, texts):
        results = self.sentiment_pipeline(texts, batch_size=8)
        return [result['label'] for result in results]

# Class to handle regard analysis
class RegardClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained("sasha/regardv3")
        model = AutoModelForSequenceClassification.from_pretrained("sasha/regardv3")
        if device >= 0:
            model.to(f'cuda:{device}')
        self.regard_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, device=device)

    def compute_regard(self, texts):
        results = self.regard_pipeline(texts, batch_size=8)
        return [result['label'] for result in results]

# Function to analyse sentiment and regard
def analyse_sentiment_and_regard(df, text_col):
    sentiment_classifier = SentimentClassifier()
    regard_classifier = RegardClassifier()

    # Compute sentiment and regard
    df['Sentiment'] = [sentiment_classifier.compute_sentiment([text])[0] for text in
                       tqdm(df[text_col], desc="Analyzing Sentiment")]
    df['Regard'] = [regard_classifier.compute_regard([text])[0] for text in tqdm(df[text_col], desc="Analyzing Regard")]

    # Output the DataFrame with new columns
    return df

# Execute on MGSD Expanded
df_mgsd_expanded = load_data("MGSD - Expanded.csv")
results_mgsd_expanded = analyse_sentiment_and_regard(df = df_mgsd_expanded, text_col = 'text')
results_mgsd_expanded.to_csv("MGSD_Expanded_sentiment_regard.csv", index=False)


