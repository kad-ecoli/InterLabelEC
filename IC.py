#!/usr/bin/python2
docstring='''
IC.py train_terms.tsv IA.txt
    calculate information content (a.k.a. information accretion)
for GO terms

Input:
    train_terms.tsv - EC number annotations
                    [1] accession
                    [2] number of EC digit
                    [3] EC number

Output
    IA.txt        - information content
                    [1] EC number, sorted in numerical order
                    [2] information content, calculated by
                        IA(q)=-log2((1+N(q))/(1+N_p(q))
                        Here, N(q) is the number of accession
                        with GO term q; N_p(q) is the number of
                        accession with all parents of GO term q
                    [3] number of training proteins
                    [4] naive probability
'''
import sys
from math import log
ln2log2=1./log(2)

def read_annotation(dbfile):
    GOdict=dict()
    target_list=[]
    ECnumber_list=[]
    fp=open(dbfile)
    for line in fp.read().splitlines():
        if line.startswith('EntryID'):
            continue
        accession,Aspect,ECnumber=line.split('\t')
        target_list.append(accession)
        if not ECnumber in GOdict:
            GOdict[ECnumber]=[accession]
            ECnumber_list.append(ECnumber)
        else:
            GOdict[ECnumber].append(accession)
    fp.close()
    GOdict['-.-.-.-']=list(set(target_list))
    isa_dict=dict()
    isa_dict['-.-.-.-']='-.-.-.-'
    for ECnumber in ECnumber_list:
        EC_list=ECnumber.split('.')
        for i in [3,2,1,0]:
            if EC_list[i]!='-':
                EC_list[i]='-'
                break
        isa_dict[ECnumber]='.'.join(EC_list)
    ia_dict=dict()
    for ECnumber in GOdict:
        ia_dict[ECnumber]=0
        if ECnumber=='-.-.-.-' or len(isa_dict[ECnumber])==0:
            continue
        NumECnumber=0
        if ECnumber in GOdict:
            NumECnumber=len(GOdict[ECnumber])
        NumParent=len(GOdict[isa_dict[ECnumber]])
        ia_dict[ECnumber]=-log((1.+NumECnumber)/(1.+NumParent))*ln2log2
    return ia_dict,GOdict

def IC(dbfile,outfile):
    ia_dict,GOdict=read_annotation(dbfile)

    txt=''
    #txt+='#ECnumber\tInformationContent\tCount\tProbability\n'
    N=len(GOdict['-.-.-.-'])
    for ECnumber in sorted(ia_dict.keys()):
        #if ECnumber=='-.-.-.-':
            #continue
        line="%s\t%.15f"%(ECnumber,ia_dict[ECnumber])
        if line.endswith("-0.000000000000000"):
            line="%s\t0.000000000000000"%(ECnumber)
        txt+=line+'\t%.9f\n'%(1.*len(GOdict[ECnumber])/N)
    fp=open(outfile,'w')
    fp.write(txt)
    fp.close()
    return

if __name__=="__main__":
    if len(sys.argv)!=3:
        sys.stderr.write(docstring)
        exit()

    dbfile  =sys.argv[1]
    outfile =sys.argv[2]
    IC(dbfile,outfile)
