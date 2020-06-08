from zipfile import ZipFile
from urllib.request import urlopen
from shutil import rmtree
from io import BytesIO
from os import path

experiments_dir = path.join(*(path.split(__file__)[:-2] + ('experiments',)))
rmtree(experiments_dir)

resp = urlopen('https://surfdrive.surf.nl/files/index.php/s/QYQTxW0bScR4XnX/download')
ZipFile(BytesIO(resp.read())).extractall(experiments_dir)

