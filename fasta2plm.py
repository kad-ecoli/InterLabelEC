#!/usr/bin/env python3
docstring='''
fasta2esm.py train_seq.fasta  embeddings/

Input:
    train_seq.fasta - fasta format sequence to embed

Output:
    embeddings/*.npy - pickle of the following dictionary
                       result['name']=protein name
                       result['mean'][34]=esmc last but two layer
                       result['mean'][35]=esmc last but two layer
                       result['mean'][36]=esmc last but two layer
                       all esmc embedding are average pooled
                       in np.array(dtype=float32) format
'''
import os, sys
import torch
import numpy as np
from settings import settings_dict as settings
from settings import training_config
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

def fasta2esm(
        fasta_file,
        output_dir,
        repr_layers:list=training_config['repr_layers'], # esmc-600m-2024-12 has 36 layers
    ):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    name_list=[]
    sequence_list=[]
    fp=open(fasta_file)
    for block in ('\n'+fp.read().strip()).split('\n>'):
        if not block.strip():
            continue
        lines=block.splitlines()
        name_list.append(lines[0].split()[0])
        sequence_list.append(''.join(lines[1:]))
    fp.close()
    
    Nseq=len(sequence_list)
    print('Embedding %d sequence from %s to %s'%(Nseq,fasta_file,output_dir))
    
    device='cpu'
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device='cuda'
    
    print("Using "+device)

    client = ESMC.from_pretrained("esmc_600m").to(device)
    EMBEDDING_CONFIG = LogitsConfig(
        #sequence=True, 
        #return_embeddings=True, 
        return_hidden_states=True
    )
    for n in tqdm(range(Nseq)):
        name=name_list[n]
        out_file = os.path.join(output_dir, name+".npy")
        if os.path.isfile(out_file):
            continue
        #print(out_file)
        result={'name':name,'mean':dict()} # 1152
        sequence=sequence_list[n][:2048]
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
        hidden_states = logits_output.hidden_states.mean(axis=2)
        for layer in repr_layers:
            result['mean'][layer]=hidden_states[layer-1][0].cpu(
                ).to(dtype=torch.float32).numpy()
        np.save(out_file, result, allow_pickle=True)
    return

def extract_embeddings(fasta_file):
    fasta2esm( fasta_file, settings['embedding_dir'])
    return

def run_mmseqs(fasta_file:str, fasta_dict:dict, db:str, output_dir:str):
    # taln, tseq
    outfile=os.path.join(output_dir,os.path.basename(fasta_file)+".m6")
    #if not os.path.isfile(outfile):
    if True:
        cmd=settings['mmseqs']+' easy-search --max-seqs 10 -s 5.7 --split-memory-limit 30G '+fasta_file+' '+db+' '+outfile+' '+settings['tmp_dir']+' --format-output query,target,qlen,tlen,alnlen,fident,bits,tseq,tstart --threads 1 -e 1e-2'
        #cmd=settings['mmseqs']+' easy-search -s 5.7 --split-memory-limit 30G '+fasta_file+' '+db+' '+outfile+' '+settings['tmp_dir']+' --format-output query,target,qlen,tlen,alnlen,fident,bits,tseq,tstart --threads 1 -e 1e-2'
        print(cmd)
        os.system(cmd)

    homolog_dict=dict()
    fp=open(outfile)
    for line in fp.read().splitlines():
        query,target,qlen,tlen,alnlen,fident,bits,tseq,tstart=line.split('\t')
        #if tseq==fasta_dict[query]:
            #continue # remove identical
        ID=float(fident)*int(alnlen)/max((int(qlen),int(tlen)))
        if not query in homolog_dict:
            homolog_dict[query]=[]
        elif len(homolog_dict[query])>=4:
        #elif len(homolog_dict[query])>=100:
            continue
        L=len(tseq)
        if L>2048:
            tstart=int(tstart)
            print("db sequence %s (L=%d) too long"%(target,L))
            if tstart+2048<=L:
                tseq=tseq[(tstart-1):(tstart+2047)]
            else:
                tseq=tseq[-2048:]
                
        homolog_dict[query].append((
            ID,
            float(bits),
            tseq.replace('-','')[:2048]
            ))
    fp.close()
    return homolog_dict

def homolog2esm(
        fasta_file:str,
        output_dir:str,
        db:str,
        repr_layers:list=training_config['repr_layers'], # esmc-600m-2024-12 has 36 layers
    ):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('running mmseqs')
    
    name_list=[]
    sequence_list=[]
    fasta_dict=dict()
    fp=open(fasta_file)
    for block in ('\n'+fp.read().strip()).split('\n>'):
        if not block.strip():
            continue
        lines=block.splitlines()
        name_list.append(lines[0].split()[0])
        sequence_list.append(''.join(lines[1:]))
        fasta_dict[name_list[-1]]=sequence_list[-1]
    fp.close()
    
    homolog_dict=run_mmseqs(fasta_file, fasta_dict, db, output_dir)
    
    Nseq=len(sequence_list)
    print('Embedding %d sequence from %s to %s'%(Nseq,fasta_file,output_dir))
    
    device='cpu'
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device='cuda'
    
    print("Using "+device)

    client = ESMC.from_pretrained("esmc_600m").to(device)
    EMBEDDING_CONFIG = LogitsConfig(
        #sequence=True, 
        #return_embeddings=True, 
        return_hidden_states=True
    )
    for n in tqdm(range(Nseq)):
        name=name_list[n]
        out_file = os.path.join(output_dir, name+".npy")
        if os.path.isfile(out_file):
            continue
        #print(out_file)
        result={'name':name,'mean':dict(),'homo':dict()} # 1152
        sequence=sequence_list[n][:2048]
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
        hidden_states = logits_output.hidden_states.mean(axis=2)
        for layer in repr_layers:
            result['mean'][layer]=hidden_states[layer-1][0].cpu(
                ).to(dtype=torch.float32).numpy()
        if not name in homolog_dict or len(homolog_dict[name])==0:
        #if False:
            for layer in repr_layers:
                result['homo'][layer]=result['mean'][layer]
        else:
            for layer in repr_layers:
                result['homo'][layer]=np.zeros(len(result['mean'][layer]))
            weight_list=[]
            if name in homolog_dict:
                for ID,bits,sequence in homolog_dict[name]:
                    weight=ID*bits
                    weight_list.append(weight)
                    protein = ESMProtein(sequence=sequence)
                    protein_tensor = client.encode(protein)
                    logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
                    hidden_states = logits_output.hidden_states.mean(axis=2)
                    for layer in repr_layers:
                        result['homo'][layer]+=(hidden_states[layer-1][0].cpu(
                            ).to(dtype=torch.float32).numpy())*weight
            if len(weight_list):
                weight=sum(weight_list)
                for layer in repr_layers:
                    result['homo'][layer]/=weight
        np.save(out_file, result, allow_pickle=True)
    return

def extract_homolog_embeddings(fasta_file, db):
    homolog2esm( fasta_file, settings['embedding_dir'], db)
    return

if __name__=="__main__":
    if len(sys.argv)!=3:
        sys.stderr.write(docstring)
        exit()

    fasta_file=sys.argv[1]
    output_dir=sys.argv[2]
    fasta2esm(fasta_file,output_dir)
