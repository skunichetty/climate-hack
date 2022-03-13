import torch
from infra.model_base import ModelBase
from infra.train import Metrics, Evaluator
from infra.dataset import AutoencoderDataset
from infra.clstm import Autoencoder, ResidualAutoencoder
from pytorch_msssim import MS_SSIM
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_Loss, self).forward(img1, img2)


num_epochs = 10
train_examples = 100
test_examples = 100

training_data = AutoencoderDataset(
    "./temp/processed",
    torch.device("cpu"),
    count=train_examples,
    train_mode=True,
)
testing_data = AutoencoderDataset(
    "./temp/processed",
    torch.device("cpu"),
    count=test_examples,
    train_mode=False,
)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=8)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=8)

# loss_fn = MS_SSIM_Loss(data_range=1.0, channel=1, win_size=7, size_average=True)
loss_fn = torch.nn.MSELoss()
model = ResidualAutoencoder(checkpoint_dir="./checkpoint/rautoencoder")
optim = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
metrics = Metrics()


def train(
    model: ModelBase,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    train: torch.utils.data.DataLoader,
    validation: torch.utils.data.DataLoader,
    metrics: Metrics,
    num_epochs,
    verbose=True,
):
    predictor = lambda x: torch.argmax(x, dim=1)
    evaluator = Evaluator(model, loss_fn, optimizer, predictor, metrics)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)
    for epoch in range(num_epochs):
        if verbose:
            print("═" * 10 + f" Epoch {epoch+1} " + "═" * 10)
            print("Training pass: ", end="")
            train_loss, _ = evaluator.train_epoch(train, verbose)
            print("Validation pass: ", end="")
            val_loss, _ = evaluator.validate(validation, verbose)
        if verbose:
            print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
        if epoch % 5 == 0:
            model.save(name=f"epoch{epoch}")
        scheduler.step()
    return (
        evaluator.train_loss,
        evaluator.train_metrics,
        evaluator.val_loss,
        evaluator.val_metrics,
    )


print(model)

train_loss, _, val_loss, _ = train(
    model, loss_fn, optim, train_loader, test_loader, metrics, num_epochs
)
model.save(name="finalepoch")

with open("./train_loss.pickle", "wb") as fout:
    pickle.dump(train_loss, fout)

with open("./val_loss.pickle", "wb") as fout:
    pickle.dump(val_loss, fout)

sns.set_theme()

batch_size = 64
train_freq = train_examples // batch_size + 1
test_freq = test_examples // batch_size + 1

sns.lineplot(x=torch.arange(0, num_epochs), y=train_loss[::train_freq])
sns.lineplot(x=torch.arange(0, num_epochs), y=val_loss[::test_freq])
plt.savefig("./loss_curves")
