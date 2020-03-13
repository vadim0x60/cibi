import numpy as np


from cibi.junior_developer import JuniorDeveloper
from cibi import bf

import logging
import logging.handlers
logger = logging.getLogger(f'cibi')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dev = JuniorDeveloper()
program_pool = [
    "+.>4ae.",
    "+.e.4e",
    "2.c-[4c,][4][]+>e3-",
    "c1,>[d,+..2]e<"
]
program_pool = [bf.Program(p) for p in program_pool]
print([p.code for p in dev.develop(program_pool, np.ones(4))])