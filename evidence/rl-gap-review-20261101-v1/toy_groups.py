# Independent re-derivation: groups needed for |z|=2 on the collision direction.
# Policy a~N(0,1), mean parameter theta; score = a. Collision iff a>t, P(C)=p.
# R=-10C+eps. Signal = mean over a group of A_i*score_i. Positive-theta raises collisions,
# so the correct direction is negative: z=sqrt(N)*|mean(g)|/sd(g), N=(2*sd/|mean|)^2.
import numpy as np
from statistics import NormalDist
rng=np.random.default_rng(20261101)
M=4_000_000
def run(p,G,noise,std_on):
    M_g=M//G
    t=NormalDist().inv_cdf(1-p)
    a=rng.standard_normal((M_g,G))
    r=-10.0*(a>t)+noise*rng.standard_normal((M_g,G))
    c=r-r.mean(1,keepdims=True)
    A=c/(r.std(1,keepdims=True)+1e-6) if std_on else c
    g=(A*a).mean(1)
    m,s=g.mean(),g.std()
    return (2*s/abs(m))**2
print(f"{'p':>5} {'G':>3} {'noise':>6} {'centered':>9} {'standardized':>13}")
for p,G in [(.08,2),(.02,2),(.08,6),(.02,6)]:
    for noise in (0.001,0.1):
        print(f"{p:5.2f} {G:3d} {noise:6.3f} {run(p,G,noise,False):9.0f} {run(p,G,noise,True):13.0f}")
print('exact ties, p=.08 G=2:', round(run(.08,2,0.0,True)))
r=np.array([-0.213,-0.214],dtype=np.float32); print('A 0.1m gap',(r-r.mean())/(r.std()+1e-6))
r=np.array([1,-9],dtype=np.float32); print('A collision',(r-r.mean())/(r.std()+1e-6))
for d in (1e-4,1e-5): print(d, d/(d+2e-6))
r=np.array([0,0,0,0,0,-10.]); print('G6 one collision',((r-r.mean())/(r.std()+1e-6)).min())
r=np.array([.4001,.4002,.3999,.4,.4003,.3998]); A=(r-r.mean())/(r.std()+1e-6); print('G6 noise rms,max',np.sqrt((A**2).mean()),np.abs(A).max())
