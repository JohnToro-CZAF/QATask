import os
import sys
from pathlib import PosixPath

if sys.version_info < (3, 5):
    raise RuntimeError('DrQA supports Python 3.5 or higher.')

DATA_DIR = (
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'qatask/database')
)

from . import tokenizers
# from . import reader
from . import retriever
# from . import pipeline