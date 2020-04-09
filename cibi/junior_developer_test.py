import numpy as np


from cibi.junior_developer import JuniorDeveloper
from cibi.codebase import Codebase
from cibi import bf

import logging
import logging.handlers
logger = logging.getLogger(f'cibi')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dev = JuniorDeveloper()
codebase = Codebase(metrics=['test_quality'])
for program in [
    "+.>4ae.",
    "+.e.4e",
    "2.c-[4c,][4][]+>e3-",
    "c1,>[d,+..2]e<"
]:
    codebase.commit(program, metrics={'test_quality': 1})
print(dev.write_programs(codebase)['code'])