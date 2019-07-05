#!/usr/bin/env bash
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 64 --ngf 16 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 64 --ngf 32 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 64 --ngf 16 --ndf 32 --lr 0.0002 &
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 64 --ngf 32 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 128 --ngf 16 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 128 --ngf 32 --ndf 16 --lr 0.0002 &
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 128 --ngf 16 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 4 --batchSize 128 --ngf 32 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 64 --ngf 16 --ndf 16 --lr 0.0002 &
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 64 --ngf 32 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 64 --ngf 16 --ndf 32 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 64 --ngf 32 --ndf 32 --lr 0.0002 &
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 128 --ngf 16 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 128 --ngf 32 --ndf 16 --lr 0.0002 &&
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 128 --ngf 16 --ndf 32 --lr 0.0002 &
python3 cnn_gan_mic.py --niter 50 --nz 8 --batchSize 128 --ngf 32 --ndf 32 --lr 0.0002
