# Mesh sanity check of a cylinder_fem_amr VTK: conformity (hanging edges), orientation, angles, levels.
# Usage: python3 -I check_mesh.py <file.vtk> [plot.png]
import sys, numpy as np, collections
f=open(sys.argv[1]).read().split('\n')
i=0
while not f[i].startswith('POINTS'): i+=1
n=int(f[i].split()[1]); P=np.array([list(map(float,f[i+1+k].split()))[:2] for k in range(n)]); i+=n+1
while not f[i].startswith('CELLS'): i+=1
m=int(f[i].split()[1]); T=np.array([list(map(int,f[i+1+k].split()))[1:] for k in range(m)]); i+=m+1
while not f[i].startswith('SCALARS amr_level'): i+=1
L=np.array([int(f[i+2+k]) for k in range(m)])
print("nodes",n,"elems",m)
print("levels",collections.Counter(L.tolist()))
a=P[T[:,1]]-P[T[:,0]]; b=P[T[:,2]]-P[T[:,0]]
area=0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
print("min area",area.min(),"neg",(area<=0).sum())
E=collections.Counter()
for t in T:
    for k in range(3):
        e=tuple(sorted((t[k],t[(k+1)%3]))); E[e]+=1
print("edge multiplicity",collections.Counter(E.values()))
# boundary edges: count==1 -> check they lie on true boundary
bad=0
xc,yc,R=0.2,0.2,0.05
def onb(p):
    return abs(p[1])<1e-6 or abs(p[1]-0.41)<1e-6 or abs(p[0])<1e-6 or abs(p[0]-2.2)<1e-6 or abs(np.hypot(p[0]-xc,p[1]-yc)-R)<1e-4
hang=0
for e,c in E.items():
    if c==1:
        mp=(P[e[0]]+P[e[1]])/2
        if not (onb(P[e[0]]) and onb(P[e[1]])): hang+=1
print("boundary edges not on physical boundary (hanging/nonconforming):",hang)
# angles
def ang(u,v): return np.degrees(np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1,1)))
A0=ang(P[T[:,1]]-P[T[:,0]],P[T[:,2]]-P[T[:,0]]);A1=ang(P[T[:,2]]-P[T[:,1]],P[T[:,0]]-P[T[:,1]]);A2=180-A0-A1
mn=np.minimum(np.minimum(A0,A1),A2)
print("min angle",mn.min(),"pct<10deg",(mn<10).mean()*100, "pct<5", (mn<5).mean()*100)
for l in sorted(set(L)): print(" lvl",l,"min ang",mn[L==l].min(),"mean",mn[L==l].mean())
cen=P[T].mean(1)
for name,msk in [("near cyl r<0.15",np.hypot(cen[:,0]-xc,cen[:,1]-yc)<0.15),("wake x 0.25-1",(cen[:,0]>0.25)&(cen[:,0]<1)),("far x>1.5",cen[:,0]>1.5),("upstream x<0.12",cen[:,0]<0.12)]:
    print(name, "elems",msk.sum(),"maxlvl",L[msk].max() if msk.any() else None, "mean lvl",L[msk].mean() if msk.any() else None)
if len(sys.argv)>2:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(16,7))
    for a_,xl in zip(ax,[(0,2.2),(0.05,0.6)]):
        a_.triplot(P[:,0],P[:,1],T,lw=0.2,color='k'); a_.tripcolor(P[:,0],P[:,1],T,facecolors=L,alpha=0.5,cmap='viridis')
        a_.set_xlim(*xl); a_.set_aspect('equal')
    plt.savefig(sys.argv[2],dpi=130,bbox_inches='tight')
