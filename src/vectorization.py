import os
import pandas as pd
import torch
from tqdm import tqdm

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.base import BaseEstimator, TransformerMixin


class Gemma_2B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hf_login()
        
        # Set the model name for Gemma 1.1 2B model
        self.model_name = "google/gemma-1.1-2b-it"

        # Load the tokenizer corresponding to the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )

        # Set the model to evaluation mode
        self.model.eval()


    def set_seed(self):
        """Set the random seed for reproducibility."""
        torch.manual_seed(33)
        torch.cuda.manual_seed_all(33)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def hf_login(self):
        """Log in to Hugging Face using a token from environment variables."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found in environment variables. Set the 'HUGGINGFACE_TOKEN' variable.")
        else:
            print('HF Token successfully found!!')
                
        # Log in to Hugging Face using the token
        login(token=hf_token)


    def fit(self, X, y=None):
        return self


    def transform(self, X, batch_size=100):
        """Generate embeddings for the input text data."""
        # Set the seed for reproducibility
        self.set_seed()
        embeddings = [] # List to store the embeddings

        # Process the input data in batches for memory efficiency
        for start in tqdm(range(0, len(X), batch_size)):
            # Select a batch of text data
            batch = X.iloc[start:start + batch_size, 0].tolist() 

            # Tokenize the batch of text
            batch_tokenized  = self.tokenizer(batch,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=20,
                                            return_tensors='pt').to('cuda')

            # Perform inference without updating gradients
            with torch.no_grad():
                outputs = self.model(**batch_tokenized, output_hidden_states=True)

            # Extract the last hidden states from the model output
            last_hidden_states = outputs.hidden_states[-1]
            batch_word_embedding  = last_hidden_states.mean(dim=1)
            
            # Append the computed embeddings to the list
            embeddings.extend(batch_word_embedding.cpu().float().numpy())
        
        print('Embeddings have been created successfully!!')
        return pd.DataFrame(embeddings)