import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, glob, gudhi, scipy.io as sio

# ---------- shared topo (delay-embedding persistence) ----------
def topo_ts(v, npts, tau, mel):
    v=np.asarray(v,float); v=(v-v.mean())/(v.std()+1e-9)
    idx=np.linspace(0,len(v)-1,npts).astype(int); v=v[idx]
    pc=np.column_stack([v[:-tau],v[tau:]])
    st=gudhi.RipsComplex(points=pc,max_edge_length=mel).create_simplex_tree(max_dimension=2)
    st.compute_persistence(); tot=0.0
    for d in (0,1):
        p=st.persistence_intervals_in_dimension(d)
        if len(p): p=p[np.isfinite(p[:,1])]; tot+=float(np.sum(p[:,1]-p[:,0]))
    return tot

fig,axs=plt.subplots(1,3,figsize=(15,4.6))

# ---------- Panel 1: MRI (6-brain) ----------
accel=np.array([1,2,2.9,4,6,8.0]); mean=np.array([112.2,93.5,70.7,55.6,34.0,23.3]); std=np.array([14.1,11.7,6.5,4.3,2.8,2.2])
ax=axs[0]
ax.fill_between(accel,mean-std,mean+std,color="#c0392b",alpha=0.18)
ax.plot(accel,mean,"o-",color="#c0392b",lw=2,ms=7)
ax.set_title("MRI  (6 brains)\nBetti-1 vs k-space undersampling",fontsize=10)
ax.set_xlabel("acceleration x"); ax.set_ylabel("Betti-1 loop count"); ax.grid(alpha=.3)
ax.text(0.5,0.92,"monotonic, training-free, reference-free",transform=ax.transAxes,ha="center",fontsize=8,style="italic")

# ---------- Panel 2: LIGO glitches ----------
CACHE="/home/Voodooaoi/ligo/cache"; CLS=["Blip","Koi_Fish","Scattered_Light"]
S={c:[topo_ts(np.load(fn),300,3,2.0) for fn in sorted(glob.glob(f"{CACHE}/{c}_*.npy"))] for c in CLS}
def auc(x,y):
    x=np.array(x);y=np.array(y);m=(x[:,None]>y[None,:]).astype(float);m[x[:,None]==y[None,:]]=.5;return m.mean()
ax=axs[1]
for i,c in enumerate(CLS):
    xs=np.random.default_rng(i).normal(i,0.06,len(S[c]))
    ax.scatter(xs,S[c],s=26,alpha=.75,label=c.replace("_"," "))
ax.set_xticks(range(3)); ax.set_xticklabels(["Blip","Koi Fish","Scat. Light"],fontsize=9)
ax.set_title("LIGO glitches  (Gravity Spy O1)\ntopological score by class",fontsize=10)
ax.set_ylabel("persistence total"); ax.grid(alpha=.3)
ax.text(0.5,0.92,f"Blip vs Koi Fish: topo AUC {auc(S['Blip'],S['Koi_Fish']):.2f}  (amplitude 0.24)",transform=ax.transAxes,ha="center",fontsize=8,style="italic")

# ---------- Panel 3: Battery ----------
def bload(cell):
    m=sio.loadmat(f"/home/Voodooaoi/battery/{cell}.mat")[cell][0,0]["cycle"][0]; V=[];C=[]
    for c in m:
        if c["type"][0]=="discharge":
            d=c["data"][0,0]; v=np.array(d["Voltage_measured"][0],float); cap=float(d["Capacity"][0,0])
            if len(v)>40: V.append(v); C.append(cap)
    return V,np.array(C)
V,C=bload("B0005"); T=np.array([topo_ts(v,80,4,0.6) for v in V]); r=np.corrcoef(T,C)[0,1]
ax=axs[2]
ax.scatter(C,T,s=20,alpha=.7,color="#2c7fb8")
ax.set_title("Battery fade  (NASA B0005)\ntopological score vs capacity",fontsize=10)
ax.set_xlabel("capacity (Ah)"); ax.set_ylabel("persistence total"); ax.grid(alpha=.3)
ax.text(0.5,0.92,f"descriptive r = {r:.2f}   (predictive: null, honest)",transform=ax.transAxes,ha="center",fontsize=8,style="italic")

fig.suptitle("One training-free topological method  ·  three unrelated domains  ·  honest limits shown",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/home/Voodooaoi/flagship_demo/money_shot.png",dpi=140)
print("saved money_shot.png")
