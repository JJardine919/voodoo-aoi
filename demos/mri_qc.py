import nibabel as nib, numpy as np, gudhi, glob

files = ['/home/Voodooaoi/fastmri/sub-01/anat/sub-01_T1w.nii.gz'] + \
        sorted(glob.glob('/home/Voodooaoi/fastmri/ds000102/sub-*/anat/*T1w.nii.gz'))

def undersample(sl, keep):
    if keep >= 1.0: return sl
    k=np.fft.fftshift(np.fft.fft2(sl))
    n=max(2,int(k.shape[0]*keep)); c=k.shape[0]//2
    m=np.zeros(k.shape[0]); m[c-n//2:c+n//2]=1
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k*m[:,None])))

def betti1(sl):
    s0=max(1,sl.shape[0]//64); s1=max(1,sl.shape[1]//64)
    s=sl[::s0,::s1]; s=(s-s.min())/(np.ptp(s)+1e-9)
    cc=gudhi.CubicalComplex(top_dimensional_cells=-s); cc.compute_persistence()
    p=cc.persistence_intervals_in_dimension(1)
    if len(p)==0: return 0
    p=p[np.isfinite(p[:,1])]
    if len(p)==0: return 0
    return int(np.sum((p[:,1]-p[:,0])>0.1))

keeps=[1.0,0.5,0.25,0.167,0.125]
table={k:[] for k in keeps}
for f in files:
    d=nib.load(f).get_fdata()
    var=np.array([np.var(d[:,:,i]) for i in range(d.shape[2])])
    idx=np.argsort(var)[-20:]
    for k in keeps:
        table[k].append(np.mean([betti1(undersample(d[:,:,i],k)) for i in idx]))

print("subjects:", len(files))
print("accel   mean   std")
for k in keeps:
    a=np.array(table[k])
    print(f"{round(1/k,1):>4}x  {round(a.mean(),1):>6}  {round(a.std(),1):>5}")
