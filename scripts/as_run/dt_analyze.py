import csv, math, sys
S=sys.argv[1]
dts=[float(x) for x in sys.argv[3].split(",")] if len(sys.argv)>3 else [0.004,0.002,0.001,0.0005]
def stats(path):
    r=list(csv.DictReader(open(path)))
    t=[float(x['time']) for x in r]; cd=[float(x['C_D']) for x in r]; cl=[float(x['C_L']) for x in r]
    i0=next(i for i,tt in enumerate(t) if tt>=4.0)
    T,CD,CL=t[i0:],cd[i0:],cl[i0:]
    m=sum(CL)/len(CL); cr=[]
    for i in range(1,len(T)):
        a,b=CL[i-1]-m,CL[i]-m
        if a<0<=b: cr.append(T[i-1]+(T[i]-T[i-1])*(-a)/(b-a))
    st=(len(cr)-1)/(cr[-1]-cr[0])*0.1 if len(cr)>2 else float('nan')
    return dict(CDmax=max(CD),CLmax=max(CL),St=st,CDmean=sum(CD)/len(CD))
for scheme in ['bdf2','euler']:
    res={dt:stats(f"{S}/{sys.argv[2]}/{scheme}_{dt}/cylinder_fem_amr_forces.csv") for dt in dts}
    ref=res[dts[-1]]
    print(f"\n{scheme.upper()}  (reference: dt = {dts[-1]})")
    print(f"{'dt':>8} " + " ".join(f"{k:>10} {'err':>9} {'order':>5}" for k in ['CDmax','CLmax','St']))
    prev={}
    for dt in dts:
        row=f"{dt:8.4f} "
        for k in ['CDmax','CLmax','St']:
            v=res[dt][k]; e=abs(v-ref[k]) if dt!=dts[-1] else float('nan')
            o=math.log2(prev[k]/e) if k in prev and e>0 and prev[k]>0 else float('nan')
            row+=f"{v:10.5f} {e:9.2e} {o:5.2f} "
            prev[k]=e
        print(row)
