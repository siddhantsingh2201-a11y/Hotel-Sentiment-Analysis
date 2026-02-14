import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ensure nltk resources are downloaded (run once)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Applies a series of cleaning steps to the input text:
        1. Lowercasing
        2. Removing HTML tags
        3. Removing punctuation and numbers
        4. Removing stopwords
        5. Lemmatization (converting words to base form)
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove HTML tags (e.g., <br>, <div>)
        text = re.sub(r'<.*?>', '', text)

        # 3. Remove punctuation and numbers
        # Keep only alphabetic characters and spaces
        text = re.sub(r'[^a-z\s]', '', text)

        # 4. Tokenize (split into words)
        words = text.split()

        # 5. Remove Stopwords & Lemmatize
        # "running" -> "run", "better" -> "good" (context dependent, simplified here)
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]

        # Join back into a single string
        return " ".join(cleaned_words)

if __name__ == "__main__":
    # Quick test
    sample_text = "I loved the hotel! The room was running hot but the staff fixed it. 5/5 stars."
    processor = TextPreprocessor()
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {processor.clean_text(sample_text)}")