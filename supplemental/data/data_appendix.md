The datasets are shown in this directory. Limited to the upload size of cmt3, they contain 1% of all data for example. 

Each dataset contains 6 numpy file, including train.npy, test.npy, train_label.npy, test_label.npy, train_complexity and test_complexity. train.npy and test.npy are the data in training set and testing set. train_label.npy and test_label.npy are the label for training set and testing set. train_compexity.npy and test_complexity is the file size of data in training set and testing set, compressed by distinct compressor: PNG, JPEG and FLIF. 

| Dataset      | Original size | Reshape Size | Num. Class | Num. Image |
| ------------ | ------------- | ------------ | ---------- | ---------- |
| Constant     | 32x32x3       | 32x32x3      | 10         | 70000      |
| Constant28   | 28x28x1       | 28x28x1      | 10         | 70000      |
| Noise        | 32x32x3       | 32x32x3      | 10         | 70000      |
| Noise28      | 28x28x1       | 28x28x1      | 10         | 70000      |
| Omniglot     | 105x105x1     | 28x28x1      | 55         | 32415      |
| MNIST        | 28x28x1       | 28x28x1      | 10         | 60000      |
| FashionMNIST | 28x28x1       | 28x28x1      | 10         | 60000      |
| KMNIST       | 28x28x1       | 28x28x1      | 10         | 60000      |
| NOTMNIST     | 28x28x1       | 28x28x1      | 10         | 60000      |
| CIFAR-10     | 32x32x3       | 32x32x3      | 10         | 60000      |
| CIFAR-100    | 32x32x3       | 32x32x3      | 100        | 60000      |
| SVHN         | 32x32x3       | 32x32x3      | 10         | 99289      |
| CelebA       | 178x218x3     | 32x32x3      | 10         | 182732     |
| TinyImageNet | 64x64x3       | 32x32x3      | 200        | 100,000    |
| LSUN         | 64x64x3       | 32x32x3      | 10         | 10000      |
| iSUN         | 64x64x3       | 32x32x3      | 10         | 8925       |

 

