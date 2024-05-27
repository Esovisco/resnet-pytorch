from pathlib import Path

import torch
from torchsummary import summary
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from dataset import SnacksDataModule
from model import ResNet, AccuracyCallback


class PredictionsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.preds = []
        self.targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        imgs, labels = batch
        preds = pl_module.forward(imgs)  # Get the predictions by forwarding the images through the model
        preds = torch.argmax(preds, dim=1)  # Get the predicted class indices
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(labels.cpu().numpy())


def index_to_prediction(idx):
    if idx == 0:
        return 'apple'
    if idx == 1:
        return 'banana'
    if idx == 2:
        return 'cookie'
    if idx == 3:
        return 'hot dog'
    if idx == 4:
        return 'muffin'


resnet18 = ResNet(in_channels=3, repeat=[2, 2, 2, 2], use_bottleneck=False, outputs=5, optimizer_lr=0.008,
                  scheduler_step=20, scheduler_gamma=0.1)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

# resnet34 = ResNet34(3, outputs=5)
# resnet34 = ResNet(in_channels=3, repeat=[3, 4, 6, 3], use_bottleneck=False, outputs=5, optimizer_lr=0.00005, scheduler_step=25, scheduler_gamma=0.1)
resnet34 = ResNet.load_from_checkpoint("tb_logs/resnet34/version_18/checkpoints/epoch=99-step=2200.ckpt", in_channels=3,
                                       repeat=[3, 4, 6, 3], use_bottleneck=False, outputs=5)
resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

resnet50 = ResNet(in_channels=3, repeat=[3, 4, 6, 3], use_bottleneck=True, outputs=5, optimizer_lr=0.0001,
                  scheduler_step=20, scheduler_gamma=0.1)
resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# resnet101 = ResNet(in_channels=3, repeat=[3, 4, 23, 3], use_bottleneck=True, outputs=5)
# resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

data_module = SnacksDataModule(
    data_dir=Path('assets/split_dataset'),
    batch_size=16
)
data_module.setup()

print("Snacks dataset sizes:")
print("Train:", len(data_module.train_dataset))
print("Validation:", len(data_module.val_dataset))
print("Test:", len(data_module.test_dataset))

tensorboard_logger = TensorBoardLogger(
    save_dir='tb_logs_test',
    name='resnet34'
)

predictions_callback = PredictionsCallback()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=100,
    logger=tensorboard_logger,
    accumulate_grad_batches=4,
    callbacks=[predictions_callback]
)

torch.set_float32_matmul_precision('high')

trainer.test(resnet34, data_module, verbose=True)

conf_matrix = confusion_matrix(
    predictions_callback.targets,
    predictions_callback.preds,
    normalize='true'
)

conf_matrix_percentages = []

for row in conf_matrix:
    result_row = []
    for el in row:
        result_row.append(el * 100)
    conf_matrix_percentages.append(result_row)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_percentages, annot=True, cmap='Blues',
    cbar_kws={'format': '%.0f%%'},
    xticklabels=[index_to_prediction(i) for i in range(5)],
    yticklabels=[index_to_prediction(i) for i in range(5)]
)
plt.xlabel('Predykcje')
plt.ylabel('Ground Truth')
plt.suptitle('Confusion matrix', fontsize='x-large')
plt.title('Wartości zostały znormalizowane wzdłuż wierszy (ground truth)')
plt.show()

# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(resnet34, data_module, attr_name='optimizer_lr')
#
# fig = lr_finder.plot(suggest=True)
# fig.show()
