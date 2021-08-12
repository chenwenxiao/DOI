## Notes on Glow

### Run Glow Code

```bash
# Build Dockerfile
cd ~/projects/dockerfiles/glow && sudo docker build -t glow .

# Run experiment
sudo docker run \
    -v /mnt/mfs:/mnt/mfs \
    -w /mnt/mfs/users/xhw15/research/ml-workspace/3rdparty/glow \
    -e MLSTORAGE_SERVER_URI="http://mlserver.ipwx.me:7980" \
    glow \
    mlrun -g 0 -- mpiexec --allow-run-as-root -n 1 \
        python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 \
        --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

### MNIST

```bash
cd /home/xhw15/research/ml-workspace/research-projects
mlrun -g 0 -- python -m flow_next.glow train \
    --dataset.name=mnist \
    --model.depth=16 --model.levels=3 --model.hidden_conv_channels='[256,256]'


cd /home/xhw15/research/ml-workspace/research-projects
mlrun -g 1 -d 'no grayscale to rgb' -- python -m flow_next.glow train \
    --dataset.name=mnist --dataset.enable_grayscale_to_rgb=false \
    --model.depth=16 --model.levels=3 --model.hidden_conv_channels='[256,256]'

```


### Cifar10

```bash
# Make Cifar10 DataSet
cd /home/xhw15/research/ml-workspace/research-projects
python -m flow_next.prepare_data cifar10 -o ../data/flow-next/cifar10/train

# my version of glow
cd /home/xhw15/research/ml-workspace/research-projects
mlrun -- python -m flow_next.glow train \
    --dataset.mmap_dir=$(pwd)/../data/flow-next/cifar10/train

# my flow model
cd /home/xhw15/research/ml-workspace/research-projects
mlrun -g 9 -- python -m flow_next.flow_1 train \
    --dataset.mmap_dir=$(pwd)/../data/flow-next/cifar10/train
```


### Innovations

Use reconstruction loss as a pre-training technique to guide the flow network
find a good initial topology?

z = f(x)
z' = z + epsilon ~ N(0,sigma_z)
minimize |f^{-1}(z') - x|^2 / sigma_x^2

anneal sigma_z and sigma_x during training, until reach a sufficiently small number.
