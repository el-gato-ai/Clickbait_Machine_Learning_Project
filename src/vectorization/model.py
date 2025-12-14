import os
import pandas as pd
import torch
from tqdm import tqdm

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.base import BaseEstimator, TransformerMixin


class HFEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device_map: str = "auto",
        torch_dtype = torch.bfloat16,
    ):
        self.hf_login()
        self.model_name = model_name

        # Load the tokenizer corresponding to the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
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
            from dotenv import load_dotenv, find_dotenv
            _ = load_dotenv(find_dotenv())
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

    def fit_transform(self, X, y=None, batch_size=100):
        """Fit the model and transform the input data to generate embeddings."""
        self.fit(X, y)
        return self.transform(X, batch_size=batch_size)
