# Lucas McCullum
# Create a Voronoi Diagram

from matplotlib import pyplot as plt
import csv
import numpy as np
import random
from itertools import combinations 

# Import the data
with open('Voronoi_Airports.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = [float(j) for i in data for j in i]
data = np.reshape(data,(len(data)//2,2))

x = []
y = []

for i in range(len(data)):
    y.append(data[i][0])
    x.append(data[i][1])

# Voronoi algorithm goes here (Delauney Triangulation)
def sortSecond(val): 
    return val[1]  

distance_dict = {}

for i in range(len(x)):

    temp_distance_list = []

    for j in range(len(x)):

        if (x[i] != x[j]):

            distance = np.sqrt(np.power(( x[j]-x[i] ),2) + np.power(( y[j]-y[i] ),2))
            temp_distance_list.append([j,distance])
    
    temp_distance_list.sort(key = sortSecond)
    distance_dict[str(i)] = temp_distance_list#[0:2] 

#print(distance_dict)

# Graph everything
# First Figure
img = plt.imread("Voronoi_USMap.jpg")
fig, ax = plt.subplots()
ax.imshow(img,extent=[1.062*min(x), 0.689*max(x), -0.025*min(y), 1.18*max(y)])

s = [50]*(len(data)//2)

plt.figure(1)

plt.scatter(x, y, s, c="r", alpha=0.5, marker=r'$\lambda$')

# Second Figure
plt.figure(2)

img = plt.imread("Voronoi_USMap.jpg")
fig, ax = plt.subplots()
ax.imshow(img,extent=[1.062*min(x), 0.689*max(x), -0.025*min(y), 1.18*max(y)])

plt.scatter(x, y, s, c="r", alpha=0.5, marker=r'$\lambda$')

#plt.axes()
x_cen_mat = []
y_cen_mat = []

comb = list(combinations(list(range(0,len(x))), 3))

for pp in range(len(comb)):

    i = comb[pp][0]
    j = comb[pp][1]
    k = comb[pp][2]

    closest_points = distance_dict[str(i)]

    x1 = x[j]
    y1 = y[j]
    x2 = x[k]
    y2 = y[k]

    fp = np.power(x[i],2) + np.power(y[i],2)
    sp = np.power(x1,2) + np.power(y1,2)
    tp = np.power(x2,2) + np.power(y2,2)
    A = [[fp,x[i],y[i],1],[sp,x1,y1,1],[tp,x2,y2,1]]

    M11 = [[A[0][1],A[0][2],A[0][3]],[A[1][1],A[1][2],A[1][3]],[A[2][1],A[2][2],A[2][3]]]
    M12 = [[A[0][0],A[0][2],A[0][3]],[A[1][0],A[1][2],A[1][3]],[A[2][0],A[2][2],A[2][3]]]
    M13 = [[A[0][0],A[0][1],A[0][3]],[A[1][0],A[1][1],A[1][3]],[A[2][0],A[2][1],A[2][3]]]
    M14 = [[A[0][0],A[0][1],A[0][2]],[A[1][0],A[1][1],A[1][2]],[A[2][0],A[2][1],A[2][2]]]

    x_cen = 0.5*(np.linalg.det(M12)/np.linalg.det(M11))
    y_cen = -0.5*(np.linalg.det(M13)/np.linalg.det(M11))

    #rad = np.sqrt(np.power(( x[j]-x_cen ),2) + np.power(( y[j]-y_cen ),2))
    rad = np.sqrt(np.power(x_cen,2) + np.power(y_cen,2) + (np.linalg.det(M14)/np.linalg.det(M11)))

    inside = 0
    
    for l in range(len(x)):

        test_distance = np.sqrt(np.power(( x[l]-x_cen ),2) + np.power(( y[l]-y_cen ),2))

        if (test_distance <= rad):

            inside += 1

    if (inside == 0):

        x_cen_mat.append(x_cen)
        y_cen_mat.append(y_cen)
        circle = plt.Circle((x_cen,y_cen),radius=rad,fill=False,linewidth=1)
        plt.gca().add_patch(circle)

        points = [[x[i],y[i]],[x[j],y[j]],[x[k],y[k]]]
        polygon = plt.Polygon(points, color=[random.random(), random.random(), random.random()], alpha=0.6)
        plt.gca().add_patch(polygon)
        
print(len(x_cen_mat))
plt.scatter(x_cen_mat, y_cen_mat, s, c="g", alpha=0.5, marker=r'$\lambda$')
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

