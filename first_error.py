"""Where does a GIDD model break? For each difficulty band, compare how many cells the two simplest Sudoku
rules can fill (naked and hidden singles, to a fixed point) with how many cells the model fills correctly,
one per pass in confidence order, before its first mistake.

  python first_error.py checkpoints/gidd_pu0_easyhard.pth 150
"""
import sys, torch, pandas as pd, numpy as np
from sudoku import gidd as g
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor
ck_path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150
UNITS=[[r*9+c for c in range(9)] for r in range(9)]+[[r*9+c for r in range(9)] for c in range(9)]+[[(br*3+r)*9+bc*3+c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
PEERS=[set() for _ in range(81)]
for u in UNITS:
    for i in u: PEERS[i].update(u)
for i in range(81): PEERS[i].discard(i)
def propagate(p):
    """naked + hidden singles to a fixed point; returns number of unknown cells it fills"""
    grid=list(p); cand=[set(range(1,10)) if v==0 else {v} for v in grid]
    for i,v in enumerate(grid):
        if v:
            for j in PEERS[i]: cand[j].discard(v)
    filled=0; changed=True
    while changed:
        changed=False
        for i in range(81):
            if grid[i]==0 and len(cand[i])==1:
                v=next(iter(cand[i])); grid[i]=v; filled+=1; changed=True
                for j in PEERS[i]: cand[j].discard(v)
        for u in UNITS:
            for d in range(1,10):
                spots=[i for i in u if grid[i]==0 and d in cand[i]]
                if len(spots)==1 and not any(grid[i]==d for i in u):
                    i=spots[0]; grid[i]=d; cand[i]={d}; filled+=1; changed=True
                    for j in PEERS[i]: cand[j].discard(d)
    return filled
df=read_hard(HARD_TEST_PATH); r=df.rating
bins=[('C other, rating 0',(r==0)&~df.source.isin(['puzzles0_kaggle','puzzles2_17_clue'])),('D rating 1-5',(r>=1)&(r<=5)),('E rating 6-20',(r>5)&(r<=20)),('G rating 51-100',(r>50)&(r<=100))]
ck=torch.load(ck_path,map_location='cpu'); m=g.SudokuGIDD(); m.load_state_dict(ck['model']); m.eval()
print(ck_path)
for name,mask in bins:
    d=df[mask].sample(n=n,random_state=0); P=to_tensor(d.question.tolist()); S=to_tensor(d.answer.tolist())
    prop=np.array([propagate(p.tolist()) for p in P]); unk=(P==0).sum(1).numpy()
    z=torch.where(P!=0,P,torch.full_like(P,g.MASK)); given=P!=0; nunk=(~given).sum(1).clamp(min=1).float()
    first=np.full(len(P),-1); step=0
    with torch.no_grad():
        while (z==g.MASK).any():
            masked=z==g.MASK; t=(masked.sum(1).float()/nunk).clamp(1e-4,1-1e-4)
            conf,pred=m(z,t,given).softmax(-1).max(-1); conf=conf.masked_fill(~masked,-1.0)
            idx=conf.argmax(1); ok=masked.any(1); val=pred.gather(1,idx[:,None]).squeeze(1)+1
            wrong=ok&(val!=S.gather(1,idx[:,None]).squeeze(1))
            for b in np.nonzero(wrong.numpy())[0]:
                if first[b]<0: first[b]=step
            z=torch.where(ok[:,None]&(torch.arange(81)[None]==idx[:,None]),val[:,None].expand(-1,81),z); step+=1
    fe=np.where(first<0,unk,first)        # correct commits before the first wrong one
    print(f"{name:20s} unknowns {unk.mean():4.1f} | propagation fills {prop.mean():4.1f} ({(prop==unk).mean()*100:3.0f}% solved outright) | model: correct before first error {fe.mean():4.1f} | corr {np.corrcoef(prop,fe)[0,1]:.2f} | model goes further than propagation in {(fe>prop).mean()*100:3.0f}% of puzzles, by {np.maximum(fe-prop,0).mean():.1f} cells on average", flush=True)
