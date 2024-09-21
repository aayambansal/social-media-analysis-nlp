import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define the custom dataset class
class TwitterDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        tweet = self.tweets[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the BERT + LSTM model
class BertLSTMClassifier(nn.Module):
    def __init__(self, bert, lstm_hidden_size, num_labels):
        super(BertLSTMClassifier, self).__init__()
        self.bert = bert
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(last_hidden_state)
        
        # Take the last LSTM output
        output = lstm_output[:, -1, :]
        
        # Classification layer
        output = self.dropout(output)
        output = self.fc(output)
        return output

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Instantiate model
model = BertLSTMClassifier(bert_model, lstm_hidden_size=128, num_labels=2)

# Training function
def train_model(model, train_loader, val_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        val_acc, val_f1 = evaluate_model(model, val_loader)
        print(f'Epoch {epoch + 1}: Validation Accuracy: {val_acc}, Validation F1: {val_f1}')
    
    return model

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids, attention_mask)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    
    return accuracy, f1

# Example usage:
# Assuming `tweets_train`, `labels_train`, `tweets_val`, and `labels_val` are available
train_dataset = TwitterDataset(tweets_train, labels_train, tokenizer, max_len=128)
val_dataset = TwitterDataset(tweets_val, labels_val, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train the model
trained_model = train_model(model, train_loader, val_loader, epochs=5, learning_rate=2e-5)

# Evaluate on test set (not shown here)
