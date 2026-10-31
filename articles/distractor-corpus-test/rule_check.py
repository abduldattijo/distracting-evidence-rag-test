"""Parse summary files and report the largest drop per variant and whether the pre-set rule is met."""
import re, sys
for f in sys.argv[1:]:
    cond=m=None; worst=(0,None); hits=[]
    for line in open(f):
        if line.startswith("-- "): cond=line[3:].strip()
        elif line[:3] in ("bm2","emb"): m=line.strip()
        mm=re.search(r"N=(\d+): recall ([\d.]+).*change ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]",line)
        if mm and mm.group(3)!="nan":
            n,ch,hi=int(mm.group(1)),float(mm.group(3)),float(mm.group(5))
            if ch<worst[0]: worst=(ch,(cond,m,n,mm.group(4),mm.group(5)))
            if ch<=-5 and hi<0: hits.append((cond,m,n,ch))
    print(f"{f}: largest drop {worst}; rule met in {len(hits)} cells {hits[:3]}")
