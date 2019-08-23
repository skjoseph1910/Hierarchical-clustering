import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import math
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import manifold

def students(array):
    dict = {}
    another = {}
    for num, i in enumerate(array):
        if i not in dict:
            dict[i] = []
            another[i] = []
        dict[i].append('student ' + str(num + 1))
        another[i].append(num)

    return dict, another

def calcualtecohesion(dict, array):
    final = []
    for i, j in dict.items():
        list = []
        for p in j:
            list.append(array[p])
        x = 0
        y = 0
        for a in list:
            x += a[0]
            y+= a[1]
        
        x = x/len(list)
        y= y/len(list)
        if x < 0 :
            x = x*-1
        if y < 0 :
            y = y*-1
    
        cohesion = 0
        for l in list:
            if l[0] <0:
                l[0] = l[0]*-1
            if l[1] <0:
                l[1] = l[1]*-1
            cohesion += (l[0]-x)*(l[0]-x)/10
            cohesion += (l[1]-y)*(l[1]-y)/10
        final.append(cohesion)

    return final



df = pd.read_excel('hello.xlsx')
array=np.array(df)
print(array)
print(len(array))
#distance matrix
DM = pairwise_distances(array, metric = 'euclidean')

print(len(array[0]))
#dendrogram
#if you want to see the dendrogram uncomment this
'''
dists=squareform(DM)
linked=linkage(dists, "complete")
labelList = range(1, len(array)+1)
dend=dendrogram(linked, labels=labelList)
plt.axhline(y=10, color='r')
plt.title("dendrogramgang")
plt.show()
'''

#silhouette
model = AgglomerativeClustering(affinity='euclidean', n_clusters=7, compute_full_tree='auto', linkage='complete').fit(array)
avg = silhouette_score(array, model.labels_)
#print(avg)


#distance matrix to x,y coordinates to calculate cohesion
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state = 3)
result = mds.fit(DM)
coords = result.embedding_
#print(type(coords))
#clusters with students
dict, another = students(model.labels_)



cohesion = calcualtecohesion(another,coords)

count = 0
for i,j in dict.items():
    print('cluster:', j, 'cohesion:', cohesion[count])
    count +=1
print('silhouette', avg)




#to see x,y plot
'''
count =0
for i, j in another.items():
    x= []
    y = []
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'black', 'magenta']
    for p in j:
        x.append(coords[p][0])
        y.append(coords[p][1])
    plt.scatter(x, y, c = colors[count])
    count +=1
plt.show()
'''



