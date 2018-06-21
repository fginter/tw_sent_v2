import sys
for idx,line in enumerate(sys.stdin):
    line=line.strip()
    if not line:
        continue
    label,txt=line.split("\t")
    print("###C: class",label)
    print(txt)
    
    
