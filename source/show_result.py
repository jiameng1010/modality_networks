import matplotlib.pyplot as plt
import numpy as np

curve_train = np.empty(shape=(40))
curve_val = np.empty(shape=(40))
for i in range(40):
    curve_train[i] = 0
    curve_val[i] = 0
for i in range(1, 40):
    filetrain = './trained_models/model_epoch_train' + str(i) + '.npy'
    fileval = './trained_models/model_epoch_val' + str(i) + '.npy'
    loss_train = np.load(filetrain)
    loss_val = np.load(fileval)
    curve_train[i] = loss_train.item()['metric_L1_real'][0]
    curve_val[i] = loss_val[1]

plt.plot(curve_train)
plt.plot(curve_val)
plt.show()



print('\n')