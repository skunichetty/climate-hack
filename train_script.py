import torch
from infra.model_base import ModelBase
from infra.train import Metrics, Evaluator, train
from infra.convlstm import ConvLSTMCell
from infra.dataset import ClimateHackDataset
import seaborn as sns

sns.set_theme()


class Nowcaster(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Nowcaster, self).__init__(*args, **kwargs)
        self.encoder = ConvLSTMCell(1, 64, 3)
        self.downsample = torch.nn.Conv2d(64, 24, 1, padding="same")
        self.pool = torch.nn.MaxPool2d(2, 2)

    def init_hidden(self, batch_size, in_height, in_width, device):
        h = torch.zeros(
            (batch_size, 64, in_height, in_width), device=device, dtype=torch.float32
        )
        c = torch.zeros(
            (batch_size, 64, in_height, in_width), device=device, dtype=torch.float32
        )
        return h, c

    def forward(self, x):
        N, T, H, W = x.shape
        state = self.init_hidden(N, H, W, device=x.device)
        h, c = state
        for i in range(T):
            h, c = self.encoder(x[:, i : i + 1], (h, c))
        output = self.downsample(h)
        torch.cuda.empty_cache()
        return self.pool(output)


loss = torch.nn.MSELoss()
model = Nowcaster("./temp/checkpoint")
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
training_data = ClimateHackDataset(
    "./temp/processed", torch.device("cuda:0"), count=100, train=True
)
testing_data = ClimateHackDataset(
    "./temp/processed", torch.device("cuda:0"), count=100, train=False
)
model.cuda()
train_loader = torch.utils.data.DataLoader(training_data, batch_size=8)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=8)


def accuracy(y_true, y_pred, y_score):
    return torch.mean(((y_true - y_pred) == 0).to(torch.float16)).item()


metrics = Metrics()
train_loss, train_acc, val_loss, val_acc = train(
    model, loss, optimizer, train_loader, test_loader, metrics, 10
)

model.save()
