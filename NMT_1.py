import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import math
import csv

# Data loading and preprocessing
def load_data(csv_file):
    hindi_sentences, english_sentences = [], []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            hindi_sentences.append(row[0])
            english_sentences.append(row[1])
    return hindi_sentences, english_sentences

def train_bpe_tokenizer(sentences, vocab_size=30000):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(sentences, vocab_size=vocab_size, min_frequency=2)
    return tokenizer

def tokenize_and_convert(sentences, tokenizer):
    return [tokenizer.encode(sentence).ids for sentence in sentences]

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=5000, pretrained_tgt_embeddings=None):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding.from_pretrained(pretrained_tgt_embeddings) if pretrained_tgt_embeddings is not None else nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        
    def forward(self, src, tgt):
        src = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    hindi_sentences, english_sentences = load_data('hindi_english_data.csv')
    hindi_tokenizer = train_bpe_tokenizer(hindi_sentences)
    english_tokenizer = train_bpe_tokenizer(english_sentences)

    hindi_vocab = hindi_tokenizer.get_vocab()
    english_vocab = english_tokenizer.get_vocab()

    hindi_indices = tokenize_and_convert(hindi_sentences, hindi_tokenizer)
    english_indices = tokenize_and_convert(english_sentences, english_tokenizer)

    # Load GloVe embeddings
    glove_embeddings = GloVe(name='42B', dim=300)

    # Create embedding matrix for English vocabulary
    english_embedding_matrix = torch.zeros((len(english_vocab), 300))
    for word, idx in english_vocab.items():
        if word in glove_embeddings.stoi:
            english_embedding_matrix[idx] = glove_embeddings[word]
        else:
            english_embedding_matrix[idx] = torch.randn(300) * 0.1

    # Initialize model
    src_vocab_size = len(hindi_vocab)
    tgt_vocab_size = len(english_vocab)
    d_model = 300

    model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, pretrained_tgt_embeddings=english_embedding_matrix)

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=hindi_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i in range(len(hindi_indices)):
            src = torch.tensor(hindi_indices[i]).unsqueeze(1)
            tgt = torch.tensor(english_indices[i]).unsqueeze(1)
            optimizer.zero_grad()
            output = model(src, tgt[:-1, :])
            loss = criterion(output.view(-1, tgt_vocab_size), tgt[1:, :].view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
