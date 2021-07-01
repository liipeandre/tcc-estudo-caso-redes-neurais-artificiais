from numpy import loadtxt, arange, uint8, uint16, float16, dtype, array_split
from pickle import dump
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from os import system
from multiprocessing import Process
from threading import Thread
from argparse import ArgumentParser, FileType, RawTextHelpFormatter
from textwrap import dedent
from itertools import product
from filelock import FileLock