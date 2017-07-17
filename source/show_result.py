import matplotlib.pyplot as plt
import numpy as np

curve_base = np.empty(shape=(40))
curve = np.empty(shape=(40))
for i in range(40):
    curve_base[i] = 0
    curve[i] = 0
#for i in range(1, 39):
#    filetrain_baseline = './trained_models_baseline/model_epoch_train' + str(i) + '.npy'
#    filetrain = './trained_models/model_epoch_train' + str(i) + '.npy'
#    loss_baseline = np.load(filetrain_baseline)
#    loss = np.load(filetrain)
#    curve_base[i] = loss_baseline.item()['conv2d_21_metric_L1_real'][0]
#    curve[i] = loss.item()['add_6_metric_L1_real'][0]

for i in range(1, 39):
    filetrain_baseline = './trained_models_baseline/model_epoch_val' + str(i) + '.npy'
    filetrain = './trained_models/model_epoch_val' + str(i) + '.npy'
    loss_baseline = np.load(filetrain_baseline)
    loss = np.load(filetrain)
    curve_base[i] = loss_baseline[5]
    curve[i] = loss[5]

plt.plot(curve_base)
plt.plot(curve)
plt.show()



print('\n')