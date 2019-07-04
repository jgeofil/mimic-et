fc = 0
print(1500/10)

with open('run.sh', 'w+') as file:
	file.write('#!/usr/bin/env bash\n')

	for nz in [4,8,16,32]:
		for lr in [0.0005, 0.0002, '0.00005']:
			for batch in [16,32,64,128,256]:
				for ndf in [16,32,64,128,256]:
					for ngf in [16,32,64,128,256]:
						fc += 1
						line = 'python3 cnn_gan_sym.py --niter 150 --nz {} --batchSize {} --ngf {} --ndf {} --lr {} {}\n'\
							.format(nz, batch, ngf, ndf, lr, '&' if fc%150==0 else '&&')
						file.write(line)
