# with harmonic.py we try to explain how harmonic functions vary under
# different scaling factor
import torch
import pdb
import matplotlib.pyplot as plt

scale = 20
L = torch.arange(scale)
vector = (2**L)*torch.pi

x = torch.linspace(-4,4,1000)
grid = vector[:,None]*x[None,:]
sinusoids = torch.sin(grid)
plt.ion()
for i in range(sinusoids.shape[0]):
    plt.plot(x,sinusoids[i,:])
    #plt.draw()
    pdb.set_trace()

plt.show()
pdb.set_trace()
