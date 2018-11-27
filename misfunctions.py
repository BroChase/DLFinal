import numpy as np
import pandas as pd



class classifications:
    def __init__(self):
        self.Desc_test = './mass_case_description_test_set.csv'
        self.Desc_train = './mass_case_description_train_set.csv'
        pass

    # load the truth values of the data malignant or benign
    def load_data(self, mode):
        if mode == 'Test':
            df = pd.read_csv(self.Desc_test)
            # pathology as cat code
            # Benign = 0
            # Malignant = 2
            # Benign no call back = 1
            df['pathology'] = df['pathology'].astype('category').cat.codes
            # make benign no callback just benign
            df['pathology'] = np.where(df['pathology'] == 1, 0, df['pathology'])
            test = np.array(df['pathology'])

            return test

        elif mode == 'Train':
            df = pd.read_csv(self.Desc_train)
            df['pathology'] = df['pathology'].astype('category').cat.codes
            df['pathology'] = np.where(df['pathology'] == 1, 0, df['pathology'])
            test = np.array(df['pathology'])

            return test
