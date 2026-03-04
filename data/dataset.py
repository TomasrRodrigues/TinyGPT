from datasets import load_dataset
from tokenizer import Tokenizer
import torch


block_size = 128
batch_size = 32

tiny_stories = load_dataset("roneneldan/TinyStories")
wiki_text = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
subset = tiny_stories["train"][:1000]
tiny_subset = tiny_stories["train"][:50]


tiny_text = "\n".join(tiny_subset["text"])
text = "\n".join(subset["text"])


tokenizer = Tokenizer(text, method="char")
tokens = tokenizer.encode(text)
tiny_tokens = tokenizer.encode(tiny_text)



class TinyDataset:
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(x), torch.tensor(y)


n= len(tokens)
train_tokens = tokens[: int(n * 0.9)]
val_tokens = tokens[int(n * 0.9) :]

train_dataset = TinyDataset(train_tokens, block_size)
val_dataset = TinyDataset(val_tokens, block_size)

tiny_dataset = TinyDataset(tiny_tokens, block_size)

def prepare_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = prepare_dataloader(train_dataset)
val_loader = prepare_dataloader(val_dataset, shuffle=False)
tiny_loader = prepare_dataloader(tiny_dataset)

xb, yb = next(iter(train_loader))
print("Batch shapes:", xb.shape, yb.shape)
x_sample, y_sample = train_dataset[10]
print("Decoded sample x:", tokenizer.decode(x_sample))
print("Decoded sample y:", tokenizer.decode(y_sample))


def preprocessing_pipeline(text, tokenize_method="char", block_size=block_size, batch_size=batch_size, tiny_text=None):
    tokenizer = Tokenizer(text, method=tokenize_method)
    tokens = tokenizer.encode(text)

    train_tokens = tokens[: int(len(tokens) * 0.9)]
    val_tokens = tokens[int(len(tokens) * 0.9) :]

    train_dataset = TinyDataset(train_tokens, block_size)
    val_dataset = TinyDataset(val_tokens, block_size)
    
    train_loader = prepare_dataloader(train_dataset, batch_size)
    val_loader = prepare_dataloader(val_dataset, batch_size, shuffle=False)

    tiny_loader = None
    if tiny_text is not None:
        tiny_tokens = tokenizer.encode(tiny_text)
        tiny_dataset = TinyDataset(tiny_tokens, block_size)
        tiny_loader = prepare_dataloader(tiny_dataset, batch_size)
    
    return train_loader, val_loader, tiny_loader, tokenizer


print("Preprocessing pipeline test:")
train_loader, val_loader, tiny_loader, tokenizer = preprocessing_pipeline(text, tiny_text=tiny_text)
assert xb.shape == yb.shape and xb.shape[1] == block_size, "Batch shapes mismatch"
print("Train loader batch shapes OK:", xb.shape, yb.shape)
x_sample, y_sample = train_loader.dataset[10]
print("Decoded sample x:", tokenizer.decode(x_sample))
print("Decoded sample y:", tokenizer.decode(y_sample))



xb_flat = xb.reshape(-1, 1).float()
yb_flat = yb.reshape(-1)

model = torch.nn.Linear(1, tokenizer.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(5):
    optimizer.zero_grad()
    logits = model(xb_flat) 
    loss = loss_fn(logits, yb_flat)
    loss.backward()
    optimizer.step()
    print(f"Step {step} | Loss: {loss.item()}")