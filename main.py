import dataprocess

if __name__ == '__main__':
    mass_train = 'Mass-Training'
    mass_test = 'Mass-Test'
    calc_train = 'Calc-Training'
    calc_test = 'Calc-Test'


    load = dataprocess.openimages()
    load.load('train')
