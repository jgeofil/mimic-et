from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import dataset.dataset as ds
from util.plot import plot_gan
from models.layers import PrintSize

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
	parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
	parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
	parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0002')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--outf', default='./out/', help='folder to output images and model checkpoints')
	parser.add_argument('--manualSeed', type=int, help='manual seed')

	opt = parser.parse_args()
	print(opt)

	ngpu = 1

	try:
		os.makedirs(opt.outf)
	except OSError:
		pass

	if opt.manualSeed is None:
		opt.manualSeed = random.randint(1, 10000)
	print("Random Seed: ", opt.manualSeed)
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)

	dataset = ds.MimicData()
	assert dataset
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
	print(device)

	nz = int(opt.nz)
	ngf = 64
	ndf = 64
	nc = 1

	# custom weights initialization called on netG and netD
	# custom weights initialization called on netG and netD
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			nn.init.normal_(m.weight.data, 0, 0.02)
		elif classname.find('BatchNorm') != -1:
			nn.init.normal_(m.weight.data, 1, 0.02)
			nn.init.constant_(m.bias.data, 0)

	class Generator(nn.Module):
		def __init__(self):
			super(Generator, self).__init__()
			self.main = nn.Sequential(
				nn.ConvTranspose2d(nz, ngf * 4, (1, 2), (1, 1), 0, bias=False),
				nn.BatchNorm2d(ngf * 4),
				nn.ReLU(True),
				# state size. (ngf*8) x 4 x 4
				nn.ConvTranspose2d(ngf * 4, ngf * 2, (1, 2), (1, 2), 0, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.ReLU(True),
				# state size. (ngf*4) x 8 x 8
				nn.ConvTranspose2d(ngf * 2, ngf * 2, (1, 2), (1, 2), 0, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.ReLU(True),
				# state size. (ngf*2) x 16 x 16
				nn.ConvTranspose2d(ngf * 2, ngf * 2, (1, 7), (1, 2), 0, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.ReLU(True),
				# state size. (ngf*2) x 16 x 16
				nn.ConvTranspose2d(ngf * 2, ngf, (6, 1), (1, 1), 0, bias=False),
				nn.BatchNorm2d(ngf),
				nn.ReLU(True),
				# state size. (ngf) x 32 x 32
				nn.ConvTranspose2d(ngf, nc, (10, 1), (8, 1), 0, bias=False),
				nn.Tanh()
				# state size. (nc) x 64 x 64
			)

		def forward(self, input):
			output = self.main(input)
			return output

	netG = Generator().to(device)

	# Handle multi-gpu if desired
	if (device.type == 'cuda') and (ngpu > 1):
		netG = nn.DataParallel(netG, list(range(ngpu)))

	netG.apply(weights_init)

	class Discriminator(nn.Module):
		def __init__(self):
			super(Discriminator, self).__init__()
			self.main = nn.Sequential(
				# input is (nc) x 64 x 64
				nn.Conv2d(nc, ndf, (10, 1), (8, 1), 0, bias=False),
				nn.BatchNorm2d(ndf),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf) x 32 x 32
				nn.Conv2d(ndf, ndf * 2, (6, 1), (6, 1), 0, bias=False),
				nn.BatchNorm2d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(ndf * 2, ndf * 2, (1, 2), (1, 2), 0, bias=False),
				nn.BatchNorm2d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(ndf * 2, ndf * 4, (1, 4), (1, 2), 0, bias=False),
				nn.BatchNorm2d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(ndf * 4, 1, (1, 3), (1, 2), 0, bias=False),
				nn.Sigmoid()
			)

		def forward(self, input):
			output = self.main(input)
			return output.view(-1, 1).squeeze(1)

	netD = Discriminator().to(device)

	if (device.type == 'cuda') and (ngpu > 1):
		netD = nn.DataParallel(netD, list(range(ngpu)))

	netD.apply(weights_init)

	criterion = nn.BCELoss()

	fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
	real_label = 1
	fake_label = 0

	# setup optimizer
	optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	ldl = []
	lgl = []
	dxl = []
	dgz1l = []
	dgz2l = []

	for epoch in range(opt.niter):
		for i, data in enumerate(dataloader, 0):

			data = data[2]
			data.unsqueeze_(1)
			data.add_(-0.5).mul_(2)

			# train with real
			netD.zero_grad()
			real_cpu = data.to(device)
			batch_size = real_cpu.size(0)
			label = torch.full((batch_size,), real_label, device=device)

			output = netD(real_cpu)
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			# train with fake
			noise = torch.randn(batch_size, nz, 1, 1, device=device)
			fake = netG(noise)

			label.fill_(fake_label)
			output = netD(fake.detach())
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			errD = errD_real + errD_fake
			optimizerD.step()

			# Update G network: maximize log(D(G(z)))
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			output = netD(fake)
			errG = criterion(output, label)
			errG.backward()
			D_G_z2 = output.mean().item()
			optimizerG.step()

			ldl.append(errD.item())
			lgl.append(errG.item())
			dxl.append(D_x)
			dgz1l.append(D_G_z1)
			dgz2l.append(D_G_z2)

			plot_gan(ldl, lgl, dxl, dgz1l, dgz2l, opt.outf)

			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
				  % (epoch, opt.niter, i, len(dataloader),
					 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
			if i % 5 == 0:
				vutils.save_image(real_cpu,
						'%s/real_samples.png' % opt.outf,
						normalize=True,pad_value=0.5)
				fake = netG(fixed_noise)
				vutils.save_image(fake.detach().apply_(lambda x: 1 if x > 0 else 0),
						'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
						normalize=True,pad_value=0.5)


if __name__ == "__main__":
	main()
