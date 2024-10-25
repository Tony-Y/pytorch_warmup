from torchvision.datasets.utils import download_url

# A downloadable URL of resnet.py in a GitHub repo:
#     https://github.com/akamaster/pytorch_resnet_cifar10
url = 'https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/refs/heads/master/resnet.py'
md5 = '9dc255cf8dc64c8b47c2b109c5b28d07'

download_url(url, root='./', filename=None, md5=md5)
