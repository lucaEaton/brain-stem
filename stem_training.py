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

torch.manual_seed(67)
if __name__ == '__main__':
    epochs = 40
    for epoch in range(epochs):
        model_stem_splitter.train()
        total_train_loss = 0

        if epoch == 20:
            # set learning rate to 1e^-4
            pass

        for batch in training_set:
            a = 1.0  # due to (1-a), this removes the others acc,
            # completely ignoring it, assuming it will make the vocals clearly n better
            mixture, vocals, song_name = batch

            vocal_pred = model_stem_splitter(mixture)

            # ~ a * loss(vocal, channel(vocal))
            loss_vocal = a * loss_fn(vocals, vocal_pred)

            # if mixture is vocals + others, then maybe acc according to the paper is acc = mix - vocals?
            # ~ (1-a) * loss(mixture, channel(mixture))
            loss_acc = (1 - a) * loss_fn(mixture-vocals, mixture-vocal_pred)

            # a * loss(vocal, channel(vocal)) + (1-a) * loss(mixture, channel(mixture)) = loss for vocal stems
            # current weight ~ reference to paper in the Fig on Pg.3, Section 2.2
            loss_train = loss_vocal + loss_acc

            optimizer.zero_grad()
            loss_train.backwards()
            optimizer.step()

            total_train_loss += loss_train.item()

        model_stem_splitter.eval()
        with torch.inference_mode():
            model_stem_splitter.eval()
            total_test_loss = 0

            for batch in testing_set:
                a = 1.0
                mixture, vocals, song_name = batch
                vocal_test = model_stem_splitter(mixture)

                vocal_test_loss = a * loss_fn(vocals, vocal_test)
                acc_test_loss = (1 - a) * loss_fn(mixture-vocals, vocal_test)
                test_loss = vocal_test_loss + acc_test_loss

                total_test_loss += test_loss.item()

        # grabbed from documentation
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {total_test_loss:.5f} | Test loss: {total_test_loss:.5f}")
