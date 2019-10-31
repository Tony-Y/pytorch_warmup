from torchvision.datasets.utils import download_url, makedir_exist_ok

raw_folder = '.data/EMNIST/raw'

url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

makedir_exist_ok(raw_folder)

# download files
print('Downloading zip archive')
download_url(url, root=raw_folder, filename="emnist.zip", md5=md5)
