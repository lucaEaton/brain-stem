import warnings

# gpt gave me this to jus clear the logs for now
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed.*"
)

warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load.*"
)

from torch.utils.data import DataLoader
from model import UNet
from dataset.dataset import musdb18
import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# out_c = 1 as for tests' sake we will be training for vocal stems.
model_stem_splitter = UNet(in_c=1, out_c=1)
model_stem_splitter = model_stem_splitter.to(device)  # either cuda (for derek) or mps(for me(luca))

# FOR DEREK :
# later within training loop I believe we (1/4)
# use this loss function to calc predicted n target (2/4)
# in order to calc total loss as displayed (3/4)
# in the Fig on Pg.3, Section 2.2 (4/4)
loss_fn = nn.L1Loss()

# learning rate = 1e^-3 (first 20 epochs) ~ weight decay (1e^-6) Pg.3, Section 2.2
optimizer = torch.optim.Adam(model_stem_splitter.parameters(), lr=0.001, weight_decay=0.000001)

# load training data
training_data = musdb18('dataset/train')
training_set = DataLoader(training_data,
                          batch_size=8,
                          shuffle=True)

# load testing data
testing_data = musdb18('dataset/test')
testing_set = DataLoader(testing_data,
                         batch_size=8,
                         shuffle=True)


def accuracy_tol(y_true, y_pred, tol=0.05):
    correct = (torch.abs(y_true - y_pred) < tol).float().mean()
    return (correct * 100).item()


torch.manual_seed(67)

if __name__ == '__main__':
    epochs = 40
    print('Training Loop Begin')
    for epoch in range(epochs):
        total_train_loss = 0
        model_stem_splitter.train()

        if epoch == 20:
            # set learning rate to 1e^-4
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

        for batch in training_set:
            a = 1.0  # due to (1-a), this removes the others acc,
            # completely ignoring it, assuming it will make the vocals clearly n better
            mixture, vocals, song_name = batch

            # mps (apple silicon) doesn't support float64 idk
            mixture = mixture.to(device, dtype=torch.float32)
            vocals = vocals.to(device, dtype=torch.float32)
            if vocals.dim() == 3:
                vocals = vocals.unsqueeze(1)
            if mixture.dim() == 3:
                mixture = mixture.unsqueeze(1)

            vocal_pred = model_stem_splitter(mixture)

            # ~ a * loss(vocal, channel(vocal))
            loss_vocal = a * loss_fn(vocals, vocal_pred)

            # if mixture is vocals + others, then maybe acc according to the paper is acc = mix - vocals?
            # ~ (1-a) * loss(mixture, channel(mixture))
            # mixture - vocal_pred < 0
            loss_acc = (1 - a) * loss_fn(mixture - vocals, mixture - vocal_pred)
            # a * loss(vocal, channel(vocal)) + (1-a) * loss(mixture, channel(mixture)) = loss for vocal stems
            # current weight ~ reference to paper in the Fig on Pg.3, Section 2.2
            loss_train = loss_vocal + loss_acc

            train_accuracy = accuracy_tol(vocals, vocal_pred)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            total_train_loss += loss_train.item()
            total_train_loss = total_train_loss / len(training_set)
        print('Eval Loop Begin')
        model_stem_splitter.eval()
        total_test_loss = 0
        with torch.inference_mode():

            for batch in testing_set:
                a = 1.0
                mixture, vocals, song_name = batch

                # mps *(apple silicon) doesn't support float64 idk
                mixture = mixture.to(device, dtype=torch.float32)
                vocals = vocals.to(device, dtype=torch.float32)
                if vocals.dim() == 3:
                    vocals = vocals.unsqueeze(1)
                if mixture.dim() == 3:
                    mixture = mixture.unsqueeze(1)

                vocal_test = model_stem_splitter(mixture)

                vocal_test_loss = a * loss_fn(vocals, vocal_test)
                acc_test_loss = (1 - a) * loss_fn(mixture - vocals, mixture - vocal_test)
                test_loss = vocal_test_loss + acc_test_loss

                test_accuracy = accuracy_tol(vocals, vocal_test)

                total_test_loss += test_loss.item()
                total_test_loss = total_test_loss / len(testing_set)

        # grabbed from documentation
        print(f"Epoch: {epoch} | Loss: {total_train_loss:.5f} | Acc: {train_accuracy:.5f} | Test loss: {total_test_loss:.5f} | Acc: {test_accuracy:.5f}")
