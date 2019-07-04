import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

def main():

    # changed configuration to this instead of argparse for easier interaction
    CUDA = False
    SEED = 1
    BATCH_SIZE = 64
    LOG_INTERVAL = 10
    EPOCHS = 25
    WORKERS = 10
    LEARN_RATE = 1e-3

    # connections through the autoencoder bottleneck
    # in the pytorch VAE example, this is 20
    ZDIMS = 30

    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)

    # DataLoader instances will load tensors directly into GPU memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    d = np.load('../../out/dims/2-diagnoses_counts.npy').astype(bool)
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
            self.fc1 = nn.Linear(M, 100)
            self.fc1a = nn.Linear(100, 100)
            # rectified linear unit layer from 400 to 400
            # max(0, x)
            self.relu = nn.ReLU()
            self.fc21 = nn.Linear(100, ZDIMS)  # mu layer
            self.fc22 = nn.Linear(100, ZDIMS)  # logvariance layer
            # this last layer bottlenecks through ZDIMS connections

            # DECODER
            # from bottleneck to hidden 400
            self.fc3 = nn.Linear(ZDIMS, 100)
            self.fc3a = nn.Linear(100, 100)
            # from hidden 400 to 784 outputs
            self.fc4 = nn.Linear(100, M)
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
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
            """THE REPARAMETERIZATION IDEA:

            For each training sample (we get 128 batched at a time)

            - take the current learned mu, stddev for each of the ZDIMS
              dimensions and draw a random sample from that distribution
            - the whole network is trained so that these randomly drawn
              samples decode to output that looks like the input
            - which will mean that the std, mu will be learned
              *distributions* that correctly encode the inputs
            - due to the additional KLD term (see loss_function() below)
              the distribution will tend to unit Gaussians

            Parameters
            ----------
            mu : [128, ZDIMS] mean matrix
            logvar : [128, ZDIMS] variance matrix

            Returns
            -------

            During training random sample from the learned ZDIMS-dimensional
            normal distribution; during inference its mean.

            """

            if self.training:
                # multiply log variance with 0.5, then in-place exponent
                # yielding the standard deviation
                std = logvar.mul(0.5).exp_()  # type: Variable
                # - std.data is the [128,ZDIMS] tensor that is wrapped by std
                # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
                #   and stddev 1 normal distribution that is 128 samples
                #   of random ZDIMS-float vectors
                eps = Variable(std.data.new(std.size()).normal_())
                #eps = Variable(std.data.new(std.size()).cauchy_())
                # - sample from a normal distribution with standard
                #   deviation = std and mean = mu by multiplying mean 0
                #   stddev 1 sample with desired std and mu, see
                #   https://stats.stackexchange.com/a/16338
                # - so we have 128 sets (the batch) of random ZDIMS-float
                #   vectors sampled from normal distribution with learned
                #   std and mu for the current input
                return eps.mul(std).add_(mu)

            else:
                # During inference, we simply spit out the mean of the
                # learned distribution for the current input.  We could
                # use a random sample from the distribution, but mu of
                # course has the highest probability.
                return mu

        def decode(self, z: Variable) -> Variable:
            h3 = self.relu(self.fc3a(self.relu(self.fc3(z))))
            return self.sigmoid(self.fc4(h3))

        def forward(self, x: Variable) -> (Variable, Variable, Variable):
            mu, logvar = self.encode(x.view(-1, M))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


    model = VAE()
    if CUDA:
        model.cuda()


    def loss_function(recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, M))

        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= BATCH_SIZE * M

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD

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
            recon_batch, mu, logvar = model(data)
            # calculate scalar loss
            loss = loss_function(recon_batch, data, mu, logvar)
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

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


    def test(epoch):
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
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data
            if i == 0:
              n = min(data.size(0), 32)
              # for the first 128 batch of the epoch, show the first 8 input digits
              # with right below them the reconstructed output digits
              comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, M)[:n]])
              comparison.detach().apply_(lambda x: 1 if x >= 0.5 else 0)
              save_image(comparison.data.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        print('====> Test set loss: {:.4f}'.format(test_loss))



    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

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
                   'results/sample_' + str(epoch) + '.png')

if __name__ == "__main__":
	main()