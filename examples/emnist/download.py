from torchvision.datasets.utils import download_url
import os

raw_folder = '.data/EMNIST/raw'

url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

os.makedirs(raw_folder, exist_ok=True)

# download files
print('Downloading zip archive')
download_url(url, root=raw_folder, filename="emnist.zip", md5=md5)
