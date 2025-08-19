import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
from safetensors.torch import load_file

class SentimentPredictor:
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the sentiment predictor with a trained model and tokenizer.
        """
        # Check if we're in test mode
        self.test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
        self.skip_model_load = os.environ.get('SKIP_MODEL_LOAD', 'false').lower() == 'true'
        
        if self.test_mode or self.skip_model_load:
            print("Running in test mode - skipping model loading")
            return
            
        # Use provided paths or environment variables
        if model_path is None:
            model_path = os.environ.get('MODEL_PATH', '/app/models/saved/sentiment_model')
        if tokenizer_path is None:
            tokenizer_path = os.environ.get('TOKENIZER_PATH', '/app/models/saved/sentiment_tokenizer')
        
        # Verify paths exist and list contents
        print(f"Checking model path: {model_path}")
        print(f"Checking tokenizer path: {tokenizer_path}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")
            
        print("Directory contents:")
        print(f"Model directory: {os.listdir(model_path)}")
        print(f"Tokenizer directory: {os.listdir(tokenizer_path)}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading tokenizer from: {tokenizer_path}")
        try:
            # First try loading from local files
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as e:
            print(f"Error loading tokenizer locally: {str(e)}")
            print("Attempting to load from HuggingFace...")
            # If local loading fails, try loading from HuggingFace
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            # Save the tokenizer locally for future use
            self.tokenizer.save_pretrained(tokenizer_path)
        
        print(f"Loading model from: {model_path}")
        try:
            # Load model configuration
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Model config not found at {config_path}")
            
            # Load model weights
            model_file = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(model_file):
                raise ValueError(f"Model weights not found at {model_file}")
            
            # Load the model with safetensors
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                use_safetensors=True,
                num_labels=2  # Binary classification (positive/negative)
            )
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to load from HuggingFace...")
            # If local loading fails, try loading from HuggingFace
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2  # Binary classification (positive/negative)
            )
            self.model.to(self.device)
            self.model.eval()
            # Save the model locally for future use
            self.model.save_pretrained(model_path)
    
    def predict(self, text):
        """
        Predict sentiment for a given text.
        Returns a dict with prediction details.
        """
        if self.test_mode or self.skip_model_load:
            # Return mock response in test mode
            return {
                "text": text,
                "sentiment": "neutral",
                "confidence": 50.0,
                "probabilities": {
                    "negative": 33.33,
                    "neutral": 33.33,
                    "positive": 33.33
                }
            }
            
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Map prediction to label
        label = "positive" if prediction == 1 else "negative"
        
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "negative": round(probabilities[0][0].item() * 100, 2),
                "positive": round(probabilities[0][1].item() * 100, 2)
            }
        }

# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = SentimentPredictor()
    
    # Test examples
    examples = [
        "This movie was fantastic! I really enjoyed every moment of it.",
        "The service was terrible and the food was cold.",
        "It was okay, not great but not terrible either."
    ]
    
    for example in examples:
        result = predictor.predict(example)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']}%)")
        print(f"Probabilities: Positive: {result['probabilities']['positive']}%, Negative: {result['probabilities']['negative']}%") 