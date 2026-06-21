from typing import List
from langchain_core.embeddings import Embeddings
from mlx_embeddings import load, generate

def selection_embeddings():
    what = "embeddings"
    choices_names = [
        "mlx-community/embeddinggemma-300m-4bit",
        "mlx-community/mxbai-embed-large-v1",
    ]
    choices = [
        createMLXGemmaEmbeddings,
        createMLXCompatibleEmbeddings,
    ]

    print(f"Select a {what}:")
    for index, option in enumerate(choices_names, start=1):
        print(f"[{index}] {option}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))

            if 1 <= choice <= len(choices):
                return choices_names[choice - 1], choices[choice - 1](choices_names[choice - 1])
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

class MLXGemmaEmbeddings(Embeddings):
    def __init__(self, model_id: str = "mlx-community/embeddinggemma-300m-4bit"):
        self.model, self.tokenizer = load(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        formatted_texts = [f"task: retrieval-document | text: {text}" for text in texts]
        
        encoded = self.tokenizer.batch_encode_plus(
            formatted_texts, 
            return_tensors="mlx", 
            padding=True,
            truncation=True
        )
        
        outputs = self.model(
            encoded["input_ids"], 
            attention_mask=encoded.get("attention_mask")
        )
        
        return outputs.text_embeds.tolist()

    def embed_query(self, text: str) -> List[float]:
        formatted_query = f"task: retrieval-query | query: {text}"
        
        encoded = self.tokenizer.batch_encode_plus(
            [formatted_query], 
            return_tensors="mlx", 
            padding=True,
            truncation=True
        )
        
        outputs = self.model(
            encoded["input_ids"], 
            attention_mask=encoded.get("attention_mask")
        )
        
        return outputs.text_embeds.tolist()[0]

class MLXCompatibleEmbeddings(Embeddings):
    def __init__(self, model_id: str = "mlx-community/mxbai-embed-large-v1"):
        self.model, self.tokenizer = load(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        output = generate(self.model, self.tokenizer, texts=texts)
        return output.text_embeds.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def createMLXGemmaEmbeddings(model_id: str):
    return MLXGemmaEmbeddings(model_id)

def createMLXCompatibleEmbeddings(model_id: str):
    return MLXCompatibleEmbeddings(model_id)

embeddings_map = {
    "mlx-community/embeddinggemma-300m-4bit": createMLXGemmaEmbeddings,
    "mlx-community/mxbai-embed-large-v1": createMLXCompatibleEmbeddings,
}

def get_embeddings(model_class_name: str):
    if not model_class_name or model_class_name not in embeddings_map:
        raise ValueError(f"could not auto-detect a valid embeddings model")
    
    return embeddings_map[model_class_name](model_class_name)