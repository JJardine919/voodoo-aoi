import csv, os, numpy as np, gudhi, itertools
from gwpy.timeseries import TimeSeries
DATA=os.environ.get("VOODOO_DATA","data")
LIGO=os.path.join(DATA,"ligo"); CACHE=os.path.join(LIGO,"cache")
CLASSES=['Blip','Koi_Fish','Scattered_Light']; NPER=15
os.makedirs(CACHE,exist_ok=True)
CSV=os.path.join(LIGO,"H1_O1.csv")
if not os.path.exists(CSV):
    raise SystemExit(f"Missing {CSV}. Run:  bash fetch_data.sh")
rows={c:[] for c in CLASSES}
with open(CSV) as f:
    for row in csv.DictReader(f):
        lab=row['ml_label']
        if lab in CLASSES and float(row['ml_confidence'])>0.9:
            try: snr=float(row['snr'])
            except: continue
            if 8<snr<50: rows[lab].append(float(row['event_time']))
for c in CLASSES: rows[c]=sorted(rows[c])[:NPER]
def getwin(t,lab):
    fn=os.path.join(CACHE,f'{lab}_{t:.3f}.npy')
    if os.path.exists(fn): return np.load(fn)
    d=TimeSeries.fetch_open_data('H1',t-16,t+16,verbose=False).whiten().bandpass(20,500)
    w=d.crop(t-0.5,t+0.5).value; np.save(fn,w); return w
def topo(v):
    v=np.asarray(v,float); v=(v-v.mean())/(v.std()+1e-9)
    idx=np.linspace(0,len(v)-1,300).astype(int); v=v[idx]; tau=3
    pc=np.column_stack([v[:-tau],v[tau:]])
    st=gudhi.RipsComplex(points=pc,max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    st.compute_persistence(); tot=0.0
    for d in (0,1):
        p=st.persistence_intervals_in_dimension(d)
        if len(p): p=p[np.isfinite(p[:,1])]; tot+=float(np.sum(p[:,1]-p[:,0]))
    return tot
def bw(v):
    v=np.asarray(v,float); V=np.abs(np.fft.rfft(v))**2; fr=np.fft.rfftfreq(len(v),1/4096)
    c=np.sum(fr*V)/(np.sum(V)+1e-9); return float(np.sqrt(np.sum((fr-c)**2*V)/(np.sum(V)+1e-9)))
T={c:[] for c in CLASSES}; A={c:[] for c in CLASSES}; B={c:[] for c in CLASSES}
for c in CLASSES:
    for i,t in enumerate(rows[c]):
        try:
            s=getwin(t,c); T[c].append(topo(s)); A[c].append(float(np.max(np.abs(s)))); B[c].append(bw(s))
            print(f"{c} {i+1}/{len(rows[c])}",flush=True)
        except Exception as e: print('skip',c,round(t,1),str(e)[:25],flush=True)
def auc(x,y):
    x=np.array(x);y=np.array(y)
    if len(x)==0 or len(y)==0: return float('nan')
    m=(x[:,None]>y[None,:]).astype(float); m[x[:,None]==y[None,:]]=0.5; return float(m.mean())
def ci(x,y,n=1000):
    x=np.array(x);y=np.array(y);r=np.random.default_rng(0);vals=[]
    for _ in range(n): vals.append(auc(r.choice(x,len(x)),r.choice(y,len(y))))
    return np.percentile(vals,2.5),np.percentile(vals,97.5)
print("\ncounts:",{c:len(T[c]) for c in CLASSES})
for a,b in itertools.combinations(CLASSES,2):
    lo,hi=ci(T[a],T[b])
    print(f"{a} vs {b}: topo {auc(T[a],T[b]):.3f} [{lo:.2f},{hi:.2f}]  amp {auc(A[a],A[b]):.3f}  bw {auc(B[a],B[b]):.3f}")
