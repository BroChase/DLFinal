import os
import re
import pydicom
import shutil


class openimages(object):
    def __init__(self):
        self.DIR = 'D:/ddsm/CBIS-DDSM/'
        # self.DIR = 'D:/Mass-Test_P_00016_LEFT_CC/10-04-2016-DDSM-30104/1-full mammogram images-14172'
        #self.DIR2 = 'D:/ddsmMassTest'
        self.DIR2 = 'D:/ddsmMassTrain'
        pass

    def load(self, mode):
        regex = re.compile("Mass-Training")
        # for file in os.listdir(self.DIR):
        #     print(file)
        #     # savename = re.findall(r"\D(\d{5})\D", self.DIR)[0]
        #     name = self.DIR.split('_')
        #     name2 = name[4].split('/')[0]
        #     savename = 'P_'+name[2]+'_'+name[3]+'_'+name2+'.dcm'
        #     shutil.copy2(self.DIR + '/' + file, self.DIR2 + '/' + savename)
        #     print('test')
        count = 0
        for file in os.listdir(self.DIR):
            if re.search(regex, file):
                name = file.split('_')
                # name to resave under
                savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4]
                for f0 in os.listdir(self.DIR + file):
                    for f1 in os.listdir(self.DIR + file + '/' + f0):
                        for f2 in os.listdir(self.DIR + file + '/' + f0 + '/' + f1):
                            f = str(self.DIR) + str(file) + '/' + str(f0) + '/' + str(f1) + '/' + str(f2)
                            if len(name) > 5:
                                if f2 == '000000.dcm':
                                    savename = savename + '00.dcm'
                                elif f2 == '000001.dcm':
                                    savename = savename + '01.dcm'
                            else:
                                savename = savename + '.dcm'
                            shutil.copy2(f, self.DIR2 + '/' + savename)
                            savename = name[1] + '_' + name[2] + '_' + name[3] + '_' + name[4]
                            # dicom_dir = pydicom.dcmread(f)
                            # data = dicom_dir.pixel_array
                            count = count + 1
                            print(count)
