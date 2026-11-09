import sys, numpy as np
sys.path.insert(0, sys.argv[3])
from viz import read
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.tri as mtri
d=read(sys.argv[1]); P,T=d['P'],d['T']; tri=mtri.Triangulation(P[:,0],P[:,1],T)
lv=d['amr_level']
title=sys.argv[4] if len(sys.argv)>4 else ''
fig,ax=plt.subplots(4,1,figsize=(18,16),gridspec_kw={'height_ratios':[1,1,1,1.6]})
for a in ax[:3]: a.set_aspect('equal'); a.set_xlim(0,2.2); a.set_ylim(0,0.41)
ax[0].tripcolor(tri,d['vorticity'],shading='gouraud',cmap='RdBu_r',vmin=-30,vmax=30); ax[0].set_title('vorticity')
ax[1].tripcolor(tri,facecolors=lv,cmap='viridis',vmin=0,vmax=4); ax[1].tricontour(tri,d['vorticity'],levels=[-20,-8,-3,3,8,20],colors='r',linewidths=0.6); ax[1].set_title('AMR level (0..4) + vorticity contours')
ax[2].tripcolor(tri,facecolors=np.log10(d['error_indicator']+1e-9),cmap='magma',vmin=-2,vmax=1); ax[2].set_title('log10 refinement score  (>0 refine, < log10(0.35) coarsen)')
ax[3].tripcolor(tri,facecolors=lv,cmap='viridis',vmin=0,vmax=4,alpha=0.6); ax[3].triplot(tri,lw=0.15,color='k'); ax[3].tricontour(tri,d['vorticity'],levels=[-20,-8,8,20],colors='r',linewidths=0.8)
ax[3].set_xlim(0.05,0.8); ax[3].set_ylim(0.0,0.41); ax[3].set_aspect('equal'); ax[3].set_title('mesh zoom')
fig.suptitle(title,fontsize=16,y=0.995)
plt.savefig(sys.argv[2],dpi=100,bbox_inches='tight')
