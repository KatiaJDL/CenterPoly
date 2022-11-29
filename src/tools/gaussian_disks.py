import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def gaussian(x,y, x_center, y_center, sigma):

  numerator = (x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)
  denominator = 2*sigma*sigma
  return -numerator/denominator

def display_gaussian_image(center, sigma):

  width = 100
  height = 100

  data = np.zeros((height, width))

  for k in range(len(center)):
    data += np.exp([[gaussian(i, j, center[k][0], center[k][1], sigma) for i in range(width)] for j in range(height)])

  ax = sns.heatmap(np.minimum(data, 1))

  plt.show()

  return data

if __name__ == '__main__':
  center = [(56,21), (10,34.5), (52.3, 43.8)]
  sigma = 10

  data1 = display_gaussian_image(center, sigma)

  center = [(56,21), (10,34.5), (52.7, 43.8)]

  data2 = display_gaussian_image(center, sigma)

  print(np.array_equal(data1,data2))
