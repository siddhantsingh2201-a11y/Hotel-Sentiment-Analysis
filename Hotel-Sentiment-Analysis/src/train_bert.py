import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
from tqdm import tqdm

# 1. Setup Device (GPU vs CPU)
# Checks if CUDA (Nvidia) or MPS (Mac M1/M2) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (M1/M2) acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU (Warning: Training will be slow)")

# 2. Load Data
data_path = os.path.join('data', 'processed_reviews.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('..', 'data', 'processed_reviews.csv')

df = pd.read_csv(data_path)
df['cleaned_review'] = df['cleaned_review'].fillna('')

# Sample data to speed up training for demonstration (Optional: remove this line for full training)
# df = df.sample(2000, random_state=42) 

sentences = df.cleaned_review.values
labels = df.sentiment.values

# 3. Tokenization (BERT specific)
print("Loading BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []

print("Tokenizing data...")
for sent in tqdm(sentences):
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,          # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert lists into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# 4. Split Data
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.2)

# 5. Create DataLoaders
batch_size = 16 # Reduce to 8 if you run out of memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 6. Load Pre-trained BERT Model
print("Loading BERT Model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False, 
)

model.to(device)

# 7. Optimizer & Learning Rate Scheduler
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 2 # BERT fine-tunes very quickly. 2-4 epochs is usually enough.

# 8. Training Loop
print(f"Starting training for {epochs} epochs...")

for epoch_i in range(0, epochs):
    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')
    
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        
        
        # Forward pass
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = result.loss
        logits = result.logits

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    print(f"  Average training loss: {avg_train_loss:.2f}")

    # Validation
    print("Running Validation...")
    model.eval()
    
    predictions, true_labels = [], []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = result.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    print(f"  Validation Accuracy: {accuracy_score(true_labels, predictions):.4f}")

# 9. Final Evaluation
print("\nTraining Complete. Final Evaluation on Validation Set:")
print(classification_report(true_labels, predictions))

# Save the model
output_dir = './models/bert_model/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")