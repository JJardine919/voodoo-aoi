import os, nibabel as nib, numpy as np, gudhi, glob
DATA=os.environ.get("VOODOO_DATA","data")
files=sorted(glob.glob(os.path.join(DATA,"mri","sub-*_T1w.nii.gz")))
if not files:
    raise SystemExit(f"No MRI data under {DATA}/mri/. Run:  bash fetch_data.sh")
keeps=[1.0,0.5,0.345,0.25,0.167,0.125]
def undersample(sl,keep):
    if keep>=1.0: return sl
    k=np.fft.fftshift(np.fft.fft2(sl)); n=max(2,int(k.shape[0]*keep)); c=k.shape[0]//2
    m=np.zeros(k.shape[0]); m[c-n//2:c+n//2]=1
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k*m[:,None])))
def betti1(sl):
    s0=max(1,sl.shape[0]//64); s1=max(1,sl.shape[1]//64)
    s=sl[::s0,::s1]; s=(s-s.min())/(np.ptp(s)+1e-9)
    cc=gudhi.CubicalComplex(top_dimensional_cells=-s); cc.compute_persistence()
    p=cc.persistence_intervals_in_dimension(1)
    if len(p)==0: return 0
    p=p[np.isfinite(p[:,1])]
    return int(np.sum((p[:,1]-p[:,0])>0.1)) if len(p) else 0
def sharp(sl):  # trivial baseline: gradient energy
    gy,gx=np.gradient(sl.astype(float)); return float(np.mean(np.sqrt(gx*gx+gy*gy)))
B={k:[] for k in keeps}; S={k:[] for k in keeps}
for f in files:
    d=nib.load(f).get_fdata(); var=np.array([np.var(d[:,:,i]) for i in range(d.shape[2])]); idx=np.argsort(var)[-20:]
    for k in keeps:
        us=[undersample(d[:,:,i],k) for i in idx]
        B[k].append(np.mean([betti1(u) for u in us])); S[k].append(np.mean([sharp(u) for u in us]))
print("subjects:",len(files))
print(f"{'accel':>6}{'Betti1 mean':>13}{'std':>7}   {'sharp mean':>11}{'std':>7}")
for k in keeps:
    b=np.array(B[k]); s=np.array(S[k])
    print(f"{round(1/k,1):>5}x{b.mean():>13.1f}{b.std():>7.1f}   {s.mean():>11.3f}{s.std():>7.3f}")
# normalized (fraction of full-sampling value) to compare shapes
b0=np.mean(B[1.0]); s0=np.mean(S[1.0])
print("\nnormalized to 1x (shape comparison):")
print(f"{'accel':>6}{'Betti1':>9}{'sharp':>9}")
for k in keeps:
    print(f"{round(1/k,1):>5}x{np.mean(B[k])/b0:>9.2f}{np.mean(S[k])/s0:>9.2f}")
