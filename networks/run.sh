#!/usr/bin/env bash
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 64 --ngf 64 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 64 --ngf 128 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 64 --ngf 64 --ndf 32 --lr 0.0002 &
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 64 --ngf 128 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 128 --ngf 64 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 128 --ngf 128 --ndf 16 --lr 0.0002 &
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 128 --ngf 64 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 16 --batchSize 128 --ngf 128 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 64 --ngf 64 --ndf 16 --lr 0.0002 &
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 64 --ngf 128 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 64 --ngf 64 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 64 --ngf 128 --ndf 32 --lr 0.0002 &
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 128 --ngf 64 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 128 --ngf 128 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 128 --ngf 64 --ndf 32 --lr 0.0002 &
python3 cnn_gan_sym.py --niter 100 --nz 32 --batchSize 128 --ngf 128 --ndf 32 --lr 0.0002
