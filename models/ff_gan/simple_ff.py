from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
	parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
	parser.add_argument('--nz', type=int, default=5, help='size of the latent z vector')
	parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0002')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--cuda', action='store_true', help='enables cuda')
	parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
	parser.add_argument('--netG', default='', help="path to netG (to continue training)")
	parser.add_argument('--netD', default='', help="path to netD (to continue training)")
	parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
	parser.add_argument('--manualSeed', type=int, help='manual seed')

	opt = parser.parse_args()
	print(opt)

	try:
		os.makedirs(opt.outf)
	except OSError:
		pass

	if opt.manualSeed is None:
		opt.manualSeed = random.randint(1, 10000)
	print("Random Seed: ", opt.manualSeed)
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)

	cudnn.benchmark = True

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	d = np.load('../../out/dims/2-diagnoses_counts.npy').astype(bool)
	M = d.shape[1]
	diag_counts_t = torch.Tensor(d)
	dataset = torch.utils.data.TensorDataset(diag_counts_t)


	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

	device = torch.device("cuda:0" if opt.cuda else "cpu")


	ngpu = int(opt.ngpu)
	nz = int(opt.nz)

	# custom weights initialization called on netG and netD
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Linear') != -1:
			m.weight.data.normal_(-0.5, 0.5)


	class Generator(nn.Module):
		def __init__(self, ngpu):
			super(Generator, self).__init__()
			self.ngpu = ngpu
			self.main = nn.Sequential(
				# input is Z, going into a convolution
				nn.Linear(nz, 50),
				nn.ReLU(True),
				nn.Linear(50, 50),
				nn.ReLU(True),
				nn.Linear(50, M),
				nn.Tanh(),
				# state size. (nc) x 64 x 64
			)

		def forward(self, input):
			if input.is_cuda and self.ngpu > 1:
				output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
			else:
				output = self.main(input)

			#output.detach().apply_(lambda x: 1 if x >= 0 else 0)
			return output


	netG = Generator(ngpu).to(device)
	netG.apply(weights_init)
	if opt.netG != '':
		netG.load_state_dict(torch.load(opt.netG))
	print(netG)


	class Discriminator(nn.Module):
		def __init__(self, ngpu):
			super(Discriminator, self).__init__()
			self.ngpu = ngpu
			self.main = nn.Sequential(
				# input is (nc) x 64 x 64
				nn.Linear(M, 50),
				nn.ReLU(True),
				nn.Linear(50, 50),
				nn.ReLU(True),
				nn.Linear(50, 1),
				nn.Sigmoid()
			)

		def forward(self, input):
			if input.is_cuda and self.ngpu > 1:
				output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
			else:
				output = self.main(input)

			return output.view(-1, 1).squeeze(1)


	netD = Discriminator(ngpu).to(device)
	netD.apply(weights_init)
	if opt.netD != '':
		netD.load_state_dict(torch.load(opt.netD))
	print(netD)

	criterion = nn.BCELoss()

	fixed_noise = torch.randn(opt.batchSize, nz, device=device)
	real_label = 1
	fake_label = 0

	# setup optimizer
	optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	for epoch in range(opt.niter):
		for i, data in enumerate(dataloader, 0):
			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			# train with real
			netD.zero_grad()
			real_cpu = data[0].to(device)
			batch_size = real_cpu.size(0)
			label = torch.full((batch_size,), real_label, device=device)

			output = netD(real_cpu)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(batch_size, nz, device=device)
			fake = netG(noise)
			#print(fake)

			label.fill_(fake_label)
			output = netD(fake.detach())
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			output = netD(fake)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
				  % (epoch, opt.niter, i, len(dataloader),
					 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
			if i % 100 == 0:
				vutils.save_image(real_cpu,
						'%s/real_samples.png' % opt.outf,
						normalize=True)
				fake = netG(fixed_noise)
				vutils.save_image(fake.detach().apply_(lambda x: 1 if x >= 0 else 0),
						'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
						normalize=True)

		# do checkpointing
		torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
		torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))


if __name__ == "__main__":
	main()
