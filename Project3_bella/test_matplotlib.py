import faulthandler
faulthandler.enable()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

print("Starting test", flush=True)

fig, ax = plt.subplots()
print("Figure created", flush=True)

ax.plot([1, 2, 3], [2, 4, 6])
print("Plot created", flush=True)

fig.canvas.draw()
print("Canvas rendered", flush=True)

fig.savefig("test.png")
print("Saved successfully", flush=True)