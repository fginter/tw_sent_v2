import sys

for line in sys.stdin:
    line=line.strip()
    if line.startswith("###"):
        print(line)
    else:
        tokens=line.split()
        for id,token in enumerate(tokens):
            print(id+1,token,*(["_"]*8),sep="\t")
        print()
        
