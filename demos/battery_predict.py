import os, scipy.io as sio, numpy as np, gudhi

DATA=os.environ.get("VOODOO_DATA","data")

def load(cell):
    fn=os.path.join(DATA,"battery",f"{cell}.mat")
    if not os.path.exists(fn):
        raise SystemExit(f"Missing {fn}. Run:  bash fetch_data.sh")
    m=sio.loadmat(fn)[cell][0,0]['cycle'][0]
    V=[];C=[]
    for c in m:
        if c['type'][0]=='discharge':
            d=c['data'][0,0]; v=np.array(d['Voltage_measured'][0],float)
            cap=float(d['Capacity'][0,0])
            if len(v)>40: V.append(v);C.append(cap)
    return V,np.array(C)

def topo(v):
    v=(v-v.min())/(np.ptp(v)+1e-9)
    n=len(v); idx=np.linspace(0,n-1,80).astype(int); v=v[idx]; tau=4
    pc=np.column_stack([v[:-tau],v[tau:]])
    st=gudhi.RipsComplex(points=pc,max_edge_length=0.6).create_simplex_tree(max_dimension=2)
    st.compute_persistence(); tot=0.0
    for dim in (0,1):
        p=st.persistence_intervals_in_dimension(dim)
        if len(p): p=p[np.isfinite(p[:,1])]; tot+=float(np.sum(p[:,1]-p[:,0]))
    return tot

def resid(y,X):
    A=np.column_stack([np.ones(len(y)),X])
    b,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    return y-A@b

H=10; allrt=[]; allry=[]
print("cell   n    partial_r(topo -> future capacity | cap,slope)")
for cell in ['B0005','B0006','B0007','B0018']:
    V,C=load(cell); T=np.array([topo(v) for v in V]); n=len(C)
    i=np.arange(H,n-H)
    Ci=C[i]; slope=C[i]-C[i-H]; Ti=T[i]; fut=C[i+H]
    X=np.column_stack([Ci,slope])
    ry=resid(fut,X); rt=resid(Ti,X)
    r=np.corrcoef(rt,ry)[0,1]; allrt.append(rt); allry.append(ry)
    print(f"{cell}   {len(i):>4}      {r:>6.3f}")
rt=np.concatenate(allrt); ry=np.concatenate(allry)
print("POOLED partial r:", round(np.corrcoef(rt,ry)[0,1],3))
