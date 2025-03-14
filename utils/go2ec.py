#!/usr/bin/env python3
docstring='''
go2ec.py is_a.csv ec2go go2ec.tsv

Input:
    is_a.csv  - GOterm Aspect direct indirect
    ec2go     - EC -> name ; GOterm
Output:
    go2ec.tsv - GOterm EC
'''
import sys

if len(sys.argv)!=4:
    sys.stderr.write(docstring)
    exit()

isafile  =sys.argv[1]
ec2gofile=sys.argv[2]
go2ecfile=sys.argv[3]

isa_dict=dict()
hasa_dict=dict()

fp=open(isafile,'r')
for line in fp.read().splitlines():
    GOterm,Aspect,direct,indirect=line.split('\t')
    if Aspect!='F':
        continue
    isa_dict[GOterm]=(','.join((direct,indirect))).split(',')
    for parent in isa_dict[GOterm]:
        if not parent in hasa_dict:
            hasa_dict[parent]=[GOterm]
        else:
            hasa_dict[parent].append(GOterm)
fp.close()

go2ec_dict=dict()
fp=open(ec2gofile,'r')
for line in fp.read().splitlines():
    if not line.startswith('EC:'):
        continue
    GOterm=line.split(' ; ')[-1]
    ECnumber=line.split(' > ')[0]
    for i in [4,3,2,1]:
        ECnumber='.'.join(ECnumber.split('.')[:i]+['-']*(4-i))
        if not GOterm in go2ec_dict:
            go2ec_dict[GOterm]=[ECnumber]
        else:
            go2ec_dict[GOterm].append(ECnumber)
        if not GOterm in hasa_dict:
            continue
        for child in hasa_dict[GOterm]:
            if not child in go2ec_dict:
                go2ec_dict[child]=[ECnumber]
            else:
                go2ec_dict[child].append(ECnumber)
fp.close()

txt=''
for GOterm in go2ec_dict:
    txt+="%s\t%s\n"%(GOterm,
        (','.join(list(set(go2ec_dict[GOterm])))).replace('EC:',''))
fp=open(go2ecfile,'w')
fp.write(txt)
fp.close()
