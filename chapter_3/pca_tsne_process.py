import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import Axes3D

PCA_switch = False
# load data
labels = np.load('labels.npy')

data = []  
total = 487168
class_num = 10575

# statistics
for i in range(class_num):
    data.append([i, 0, []])
for i in range(total):
    index = int(labels[i])
    data[index][1] += 1
    data[index][2].append(i)

# choose data
data = sorted(data, key=lambda x:x[1], reverse=True)
res = data[892:902]

# put data in res_picture and res_labels(0~9)
res_picture = []
res_labels = []
for i, ele in enumerate(res):
    la = ele[0]
    for p in ele[2]:
        res_picture.append([p, la, i])

res_picture = sorted(res_picture)

for ele in res_picture:
    res_labels.append(ele[2])


for checkpoint in range(11):

    ckpoint = checkpoint * 5
    feature_data = np.load('feature_'+str(ckpoint)+'.npy')
    # put choose feature in res_feature
    res_feature = []
    for ele in res_picture:
        res_feature.append(feature_data[ele[0]])

    print(len(res_picture), res_feature[0].shape)

    # decomposition
    if PCA_switch:
        pca = PCA(n_components=2)
        new_res_feature = pca.fit_transform(res_feature)
        print(new_res_feature.shape)
    else:
        tsne = TSNE(n_components=2)
        new_res_feature = tsne.fit_transform(res_feature)
        print(new_res_feature.shape)

    # 3D data
    if PCA_switch:
        pca = PCA(n_components=3)
        new_res_feature_3D = pca.fit_transform(res_feature)
        print(new_res_feature_3D.shape)
    else:
        tsne = TSNE(n_components=3)
        new_res_feature_3D = tsne.fit_transform(res_feature)
        print(new_res_feature_3D.shape)
    print('/*****************************************/')


    color_mapping = {0:"#9b59b6", 1:"#3498db", 2:"#95a5a6", 3:"#e74c3c", 4:"#34495e", 5:"#2ecc71",
                        6:"#92a916", 7:"#34182b", 8:"#e71c8c", 9:"#fe1c01"}

    # plot
    colors = list(map(lambda x: color_mapping[x], res_labels))

    plt.scatter(new_res_feature[:, 0], new_res_feature[:, 1], s=8, c=colors)
    plt.savefig('data/result_'+str(ckpoint)+'.png')
    plt.clf()
    # 
    ax = Axes3D(plt.figure())
    ax.scatter3D(new_res_feature_3D[:, 0], new_res_feature_3D[:, 1], new_res_feature_3D[:, 2], s=8, c=colors)
    plt.savefig('data/result_3D_'+str(ckpoint)+'.png')
    plt.clf()

