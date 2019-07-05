fc = 0
print(1500/6)

with open('run.sh', 'w+') as file:
	file.write('#!/usr/bin/env bash\n')

	for nz in [16,32]:
		for lr in [0.0002]:
			for batch in [64,128]:
				for ndf in [16,32]:
					for ngf in [64,128]:
						fc += 1
						line = 'python3 cnn_gan_sym.py --niter 100 --nz {} --batchSize {} --ngf {} --ndf {} --lr {} {}\n'\
							.format(nz, batch, ngf, ndf, lr, '&' if fc%3==0 else '&&')
						file.write(line)
