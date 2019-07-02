import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from matplotlib import pyplot as plt

def plot(train, test, filename):
    plt.close()
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.grid()
    plt.legend()
    plt.savefig(filename+'/loss.png')


def main():

    # changed configuration to this instead of argparse for easier interaction
    CUDA = False
    SEED = 1
    BATCH_SIZE = 64
    LOG_INTERVAL = 10
    EPOCHS = 100
    WORKERS = 10
    LEARN_RATE = 1e-3
    INFO = 'diag500'

    # connections through the autoencoder bottleneck
    # in the pytorch VAE example, this is 20
    L1DIM = 100
    L2DIM = 100
    ZDIMS = 40

    filename = '_'.join([INFO, str(SEED), str(BATCH_SIZE), str(EPOCHS), str(LEARN_RATE), str(L1DIM), str(L2DIM), str(ZDIMS)])

    if not os.path.exists(filename):
        os.makedirs(filename)

    train_l = []
    test_l = []

    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)

    # DataLoader instances will load tensors directly into GPU memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    d = np.load('../../out/dims/2-500diagnoses_counts.npy').astype(bool)
    M = d.shape[1]
    print(M)
    diag_counts_t = torch.Tensor(d)
    full_dataset = torch.utils.data.TensorDataset(diag_counts_t)


    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_d, test_d = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True, num_workers=int(WORKERS))
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=True, num_workers=int(WORKERS))

    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            # ENCODER
            # 28 x 28 pixels = 784 input pixels, 400 outputs
            self.fc1 = nn.Linear(M, L1DIM)
            self.fc1a = nn.Linear(L1DIM, L2DIM)
            # rectified linear unit layer from 400 to 400
            # max(0, x)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(L2DIM, ZDIMS)

            # DECODER
            # from bottleneck to hidden 400
            self.fc3 = nn.Linear(ZDIMS, L2DIM)
            self.fc3a = nn.Linear(L2DIM, L1DIM)
            # from hidden 400 to 784 outputs
            self.fc4 = nn.Linear(L1DIM, M)
            self.sigmoid = nn.Sigmoid()

        def encode(self, x: Variable) -> (Variable, Variable):
            """Input vector x -> fully connected 1 -> ReLU -> (fully connected
            21, fully connected 22)

            Parameters
            ----------
            x : [128, 784] matrix; 128 digits of 28x28 pixels each

            Returns
            -------

            (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
                variance units one for each latent dimension

            """

            # h1 is [128, 400]
            h1 = self.relu(self.fc1a(self.relu(self.fc1(x))))  # type: Variable
            return self.fc2(h1)

        def decode(self, z: Variable) -> Variable:
            h3 = self.relu(self.fc3a(self.relu(self.fc3(z))))
            return self.sigmoid(self.fc4(h3))

        def forward(self, x: Variable) -> Variable:
            z = self.encode(x.view(-1, M))
            return self.decode(z)

    model = VAE()
    if CUDA:
        model.cuda()

    def loss_function(recon_x, x) -> Variable:
        # how well do input x and output recon_x agree?
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, M))
        return BCE

    # Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    def train(epoch):
        # toggle model to train mode
        model.train()
        train_loss = 0
        # in the case of MNIST, len(train_loader.dataset) is 60000
        # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
        for batch_idx, data in enumerate(train_loader):
            data = data[0]
            data = Variable(data)
            if CUDA:
                data = data.cuda()
            optimizer.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch = model(data)
            # calculate scalar loss
            loss = loss_function(recon_batch, data)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.data
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data / len(data)))

        train_l.append(train_loss / len(train_loader.dataset))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    def test(epoch):
        with torch.no_grad():
            # toggle model to test / inference mode
            model.eval()
            test_loss = 0

            # each data is of BATCH_SIZE (default 128) samples
            for i, data in enumerate(test_loader):
                data = data[0]
                if CUDA:
                    # make sure this lives on the GPU
                    data = data.cuda()

                # we're only going to infer, so no autograd at all required: volatile=True
                data = Variable(data)
                recon_batch = model(data)
                test_loss += loss_function(recon_batch, data).data
                if i == 0:
                    n = min(data.size(0), 32)
                    # for the first 128 batch of the epoch, show the first 8 input digits
                    # with right below them the reconstructed output digits
                    real_d = data[:n].detach().apply_(lambda x: 1 if x >= 0.5 else 0)
                    recon_d = recon_batch.view(BATCH_SIZE, M)[:n].detach().apply_(lambda x: 1 if x >= 0.5 else 0)
                    diff_d = [[1 if l == 0 and o == 1 else 0 if l == 1 and o == 0 else 0.5
                            for l, o in zip(rl, ro)
                        ] for rl, ro in zip(real_d, recon_d)
                    ]
                    diff_d = Variable(torch.Tensor(diff_d))
                    save_image(diff_d.data.cpu(), filename+'/reconstruction_' + str(epoch) + '.png', nrow=n)

            test_l.append(test_loss / len(test_loader.dataset))

            print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

        plot(train_l, test_l, filename)

        # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space
        sample = Variable(torch.randn(64, ZDIMS))
        if CUDA:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        sample.detach().apply_(lambda x: 1 if x >= 0.5 else 0)

        # save out as an 8x8 matrix of MNIST digits
        # this will give you a visual idea of how well latent space can generate things
        # that look like digits
        save_image(sample.data.view(64, M),
                   filename+'/sample_' + str(epoch) + '.png')

if __name__ == "__main__":
	main()