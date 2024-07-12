from pylab import *

x = rand(20)
y = rand(20)

# Lines on top of scatter
figure()
subplot(211)
plot(x, y, "r", lw=3)
scatter(x, y, s=120)
