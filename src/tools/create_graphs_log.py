import matplotlib.pyplot as plt

def main():
  with open('log.txt') as f:
    lines = f.readlines()

  glob_loss = []
  hm_l = []
  off_l = []
  poly_l = []
  depth_l = []

  glob_loss_val = []
  hm_l_val = []
  off_l_val = []
  poly_l_val = []
  depth_l_val = []

  for epoch in lines:
    m = epoch.split("|")

    if m[0].split(':')[1] == ' AP':
      glob_loss_val.append(float(m[1][5:-1]))
      hm_l_val.append(float(m[2][5:-1]))
      off_l_val.append(float(m[3][6:-1]))
      poly_l_val.append(float(m[4][7:-1]))
      depth_l_val.append(float(m[5][8:-1]))

    else:
      nb_epoch = int(m[0].split(":")[-1])
      glob_loss.append(float(m[1][5:-1]))
      hm_l.append(float(m[2][5:-1]))
      off_l.append(float(m[3][6:-1]))
      poly_l.append(float(m[4][7:-1]))
      depth_l.append(float(m[5][8:-1]))

      if len(m) > 8 :
        glob_loss_val.append(float(m[7][5:-1]))
        hm_l_val.append(float(m[8][5:-1]))
        off_l_val.append(float(m[9][6:-1]))
        poly_l_val.append(float(m[10][7:-1]))
        depth_l_val.append(float(m[11][8:-1]))

  plt.plot(glob_loss, label = "glob_loss")
  plt.plot(hm_l, label = "hm_l")
  plt.plot(off_l, label = "off_l")
  plt.plot(poly_l, label = "poly_l")
  plt.plot(depth_l, label = "depth_l")
  plt.legend()
  plt.savefig("loss_train.png")
  plt.show()

  plt.figure()

  plt.plot(glob_loss_val, label = "glob_loss_val")
  plt.plot(hm_l_val, label = "hm_l_val")
  plt.plot(off_l_val, label = "off_l_val")
  plt.plot(poly_l_val, label = "poly_l_val")
  plt.plot(depth_l_val, label = "depth_l_val")
  plt.legend()
  plt.savefig("loss_valid.png")
  plt.show()



if __name__ == '__main__':
  main()
