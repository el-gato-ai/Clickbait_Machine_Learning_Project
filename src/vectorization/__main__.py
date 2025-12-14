"""
Run vectorization pipeline for the raw data and embedding parquet files.
Make sure to have the required environment variables set, e.g., in a .env file in the root.

"""

from src.vectorization.vectorize import embed_all

if __name__ == "__main__":
    # Load environment variables from a .env file if present
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
    
    # Run the embedding process on all raw data files
    embed_all()
