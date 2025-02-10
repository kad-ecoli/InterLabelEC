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

if __name__=="__main__":
    if len(sys.argv)!=3:
        sys.stderr.write(docstring)
        exit()

    fasta_file=sys.argv[1]
    output_dir=sys.argv[2]
    fasta2esm(fasta_file,output_dir)
