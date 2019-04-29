# Lucas McCullum
# Create a Voronoi Diagram

from matplotlib import pyplot as plt
import csv
import numpy as np

with open('Voronoi_Airports.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

data = [float(j) for i in data for j in i]
data = np.reshape(data,(len(data)//2,2))

x = []
y = []

for i in range(len(data)):
    y.append(data[i][0])
    x.append(data[i][1])

img = plt.imread("Voronoi_USMap.jpg")
fig, ax = plt.subplots()
ax.imshow(img,extent=[1.062*min(x), 0.689*max(x), -0.025*min(y), 1.18*max(y)])

s = [50]*(len(data)//2)
plt.scatter(x, y, s, c="r", alpha=0.5, marker=r'$\clubsuit$',
            label="Luck")

# plt.axes()
# xdat = [[2,8,8],[2,8,2]]
# ydat = [[1,1,4],[4,4,1]]
# points1 = [[2, 1], [8, 1], [8, 4]]
# points2 = [[2, 4], [8, 4], [2, 1]]
# polygon1 = plt.Polygon(points1, color=[1, 0, 0])
# polygon2 = plt.Polygon(points2, color=[0, 1, 0])
# plt.gca().add_patch(polygon1)
# plt.gca().add_patch(polygon2)
# plt.axis('scaled')
plt.show()

