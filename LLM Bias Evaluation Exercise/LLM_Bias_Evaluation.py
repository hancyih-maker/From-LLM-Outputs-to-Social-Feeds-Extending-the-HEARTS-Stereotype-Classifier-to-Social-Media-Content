import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_and_save(input_csv, model, tokenizer, device): #
    data = pd.read_csv(input_csv)
    
    texts = data['text'].tolist()
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)#
    
    results = classifier(texts, batch_size=64)
    
    data['prediction'] = [result['label'] for result in results]
    
    output_data = data[['text', 'group', 'prediction']]  
    
    output_file = input_csv.replace('.csv', '_predictions.csv')
    output_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    files_to_process = ['Claude-2 Outputs.csv', 'Claude-3.5-Sonnet Outputs.csv', 'Claude-3-Sonnet Outputs.csv', 'Gemini-1.0-Pro Outputs.csv', 'Gemini-1.5-Pro Outputs.csv', 'GPT-3.5-Turbo Outputs.csv',
			'GPT-4o Outputs.csv', 'GPT-4-Turbo Outputs.csv', 'Llama-3-70B-T Outputs.csv', 'Llama-3.1-405B-T Outputs.csv', 'Mistral Large 2 Outputs.csv', 'Mistral Medium Outputs.csv']
    
    model_name = "holistic-ai/bias_classifier_albertv2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1 #

    for filename in files_to_process:
        print(f"Processing {filename}...")
        predict_and_save(filename, model, tokenizer, device)

if __name__ == "__main__":
    main()
