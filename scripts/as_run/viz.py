import sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def read(fn):
    L=open(fn).read().split('\n'); d={}; i=0
    while i<len(L):
        s=L[i].split()
        if not s: i+=1; continue
        if s[0]=='POINTS':
            n=int(s[1]); d['P']=np.array([list(map(float,L[i+1+k].split()))[:2] for k in range(n)]); i+=n+1; continue
        if s[0]=='CELLS':
            m=int(s[1]); d['T']=np.array([list(map(int,L[i+1+k].split()))[1:] for k in range(m)]); i+=m+1; continue
        if s[0]=='VECTORS':
            n=len(d['P']); d[s[1]]=np.array([list(map(float,L[i+1+k].split()))[:2] for k in range(n)]); i+=n+1; continue
        if s[0]=='SCALARS':
            name=s[1]; i+=2; vals=[]
            while i<len(L) and L[i].strip() and not L[i][0].isalpha():
                vals.append(float(L[i])); i+=1
            d[name]=np.array(vals); continue
        i+=1
    return d
d=read(sys.argv[1]); P,T=d['P'],d['T']
fig,ax=plt.subplots(4,1,figsize=(18,15))
for a in ax: a.set_aspect('equal'); a.set_xlim(0,2.2); a.set_ylim(0,0.41)
ax[0].tripcolor(P[:,0],P[:,1],T,d['vorticity'],shading='gouraud',cmap='RdBu_r',vmin=-30,vmax=30); ax[0].set_title('vorticity')
ax[1].tripcolor(P[:,0],P[:,1],T,facecolors=d['amr_level'],cmap='viridis'); ax[1].triplot(P[:,0],P[:,1],T,lw=0.1,color='k'); ax[1].set_title('level')
ax[2].tripcolor(P[:,0],P[:,1],T,facecolors=np.log10(d['error_indicator']+1e-12),cmap='magma'); ax[2].set_title('log10 eta')
ax[3].triplot(P[:,0],P[:,1],T,lw=0.15,color='k'); ax[3].set_xlim(0.05,0.75); ax[3].set_title('mesh zoom')
plt.savefig(sys.argv[2],dpi=110,bbox_inches='tight')
