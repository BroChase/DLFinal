import os
import re


class openimages(object):
    def __init__(self):
        self.DIR = 'E:/ddsm/CBIS-DDSM/'
        pass

    def load(self, mode):
        regex = re.compile("Calc-Test")
        for file in os.listdir(self.DIR):
            if re.search(regex, file):
                for f0 in os.listdir(self.DIR + file):
                    for f1 in os.listdir(self.DIR + file + '/' + f0):
                        for f2 in os.listdir(self.DIR + file + '/' + f0 + '/' + f1):
                            print(f2)