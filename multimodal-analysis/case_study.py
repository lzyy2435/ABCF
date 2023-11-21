import numpy as np
from sklearn.decomposition import PCA
import random
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

feature_list = np.loadtxt('domain_pool_small.txt')
label_list = np.loadtxt('domain_label_small.txt')

print(feature_list.dtype)
print(feature_list.shape)

model = PCA(3)
scaler = preprocessing.StandardScaler()
feature_list = scaler.fit_transform(feature_list)
new_feature = model.fit_transform(feature_list)

print(model.singular_values_)
print(model.explained_variance_ratio_)
print(model.components_)

label_list = label_list.reshape((label_list.shape[0], 1))
new_data = np.concatenate((new_feature, label_list), axis=1)
print(new_data)

new_data = random.sample(list(new_data), 200)

x_val = [i[0] for i in new_data]
y_val = [i[1] for i in new_data]
z_val = [i[2] for i in new_data]

labels = [int(i[3]) for i in new_data]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_val, y_val, z_val, c=labels, cmap='coolwarm', marker='^')
ax.set_zlabel('axis3', fontdict={'size': 15, 'color': 'blue'})
ax.set_ylabel('axis2', fontdict={'size': 15, 'color': 'blue'})
ax.set_xlabel('axis1', fontdict={'size': 15, 'color': 'blue'})

plt.savefig('pca_stage1.png')
plt.show()

