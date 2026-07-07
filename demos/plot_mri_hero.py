import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
accel = [1.0, 2.0, 2.9, 4.0, 6.0, 8.0]
betti = [109.4, 93.0, 75.4, 61.1, 38.1, 25.8]
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(accel, betti, "o-", color="#c0392b", lw=2.2, ms=8)
for x,y in zip(accel,betti):
    ax.annotate(f"{y:.0f}", (x,y), textcoords="offset points", xytext=(6,6), fontsize=9)
ax.set_xlabel("Acceleration factor (k-space undersampling)")
ax.set_ylabel("Betti-1 loop count (topological complexity)")
ax.set_title("Training-free, reference-free MRI quality index\nBetti-1 degrades monotonically with undersampling (sub-01 T1w)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/Voodooaoi/flagship_demo/mri_hero_curve.png", dpi=140)
print("saved mri_hero_curve.png")
