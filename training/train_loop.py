from training.optimizer import get_optimizer
from model.tinygpt import TinyGPT
from training.scheduler import WarmupCosineLearningRateScheduler
import torch.nn.functional as F
from data.data_pipeline import TinyGPTDataPipeline, load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_lr = 0.1
weight_decay = 0.01
data_pipeline = TinyGPTDataPipeline(block_size=128, batch_size=32, tokenize_method="char")
data_pipeline.load_datasets(tinystories_subset=50, wikitext_subset=1000)

data_pipeline.tiny_stories = load_dataset("roneneldan/TinyStories")["train"][:1000]
#data_pipeline.tiny_subset = load_dataset("roneneldan/TinyStories")["train"][:50]

text = "\n".join(data_pipeline.tiny_stories["text"])
tiny_text = "\n".join(data_pipeline.tiny_subset["text"])

train_loader, val_loader, tiny_loader, tokenizer = data_pipeline.preprocess(text, tiny_text=tiny_text)

xb, yb = data_pipeline.sanity_check(train_loader)


model = TinyGPT(vocab_size = tokenizer.vocab_size, block_size = 128, d_model=256, n_heads=8, n_layers=6).to(device)
model.train()

opt = get_optimizer(model, lr=max_lr, weight_decay=weight_decay)
steps_per_epoch = len(train_loader)
scheduler = WarmupCosineLearningRateScheduler(
    max_lr=0.1,
    total_epochs=20,
    steps_per_epoch=steps_per_epoch,
    warmup_epochs=5,
    min_lr=0.001
)

global_step = 0
for epoch in range(20):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        lr = scheduler.get_learning_rate(global_step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        global_step += 1