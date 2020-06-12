from zipfile import ZipFile
from urllib.request import urlopen
from shutil import rmtree
from io import BytesIO
from os import path

from cibi.utils import get_project_dir

experiments_dir = get_project_dir('experiments')
rmtree(experiments_dir)

resp = urlopen('https://surfdrive.surf.nl/files/index.php/s/QYQTxW0bScR4XnX/download')
ZipFile(BytesIO(resp.read())).extractall(experiments_dir)

