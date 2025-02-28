#!/usr/bin/env python3
import pandas as pd
import os, sys, subprocess, argparse
import numpy as np
import random
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import shutil

from fasta2plm import extract_homolog_embeddings,extract_embeddings
from settings import settings_dict as settings
from IC import IC

docstring = """
example usage:
    python prepare_data.py -train Data/train_raw/train_terms.tsv -train_seqs Data/train_raw/train_seq.fasta -d Data --ia

"""


#oboTools = ObOTools()
aspects = ['EC1', 'EC2', 'EC3', 'EC4']

def get_seq_dict(fasta_file:str)->dict:
    """read fasta file and return a dict

    Args:
        fasta_file (str): path of fasta file

    Returns:
        dict: dict of fasta file, key is protein name, value is protein sequence
    """
    seq_dict = {}
    fp=open(fasta_file)
    for block in ('\n'+fp.read()).split('\n>'):
        if len(block.strip())==0:
            continue
        lines=block.splitlines()
        header=lines[0]
        sequence=''.join(lines[1:])
        seq_dict[header]=sequence
    fp.close()
    return seq_dict

def prop_parents(df:pd.DataFrame)->pd.DataFrame:
    """propagate parent terms to the dataframe

    Args:
        df (pd.DataFrame): dataframe with columns ['EntryID', 'term']

    Returns:
        pd.DataFrame: dataframe with columns ['EntryID', 'term']
    """
    aspect_df = []
    for aspect in aspects:
        cur_df = df[df['aspect'] == aspect].copy()
        cur_df = cur_df.groupby(['EntryID'])['term'].apply(set).reset_index()
        #cur_df['term'] = cur_df['term'].apply(lambda x: oboTools.backprop_terms(x))
        cur_df = cur_df.explode('term')
        cur_df['aspect'] = aspect
        aspect_df.append(cur_df)
    
    prop_df = pd.concat(aspect_df, axis=0)
    return prop_df


def goset2vec(goset:set, aspect_go2vec_dict:dict, fixed_len:bool=False):
    if fixed_len:
        num_go = len(aspect_go2vec_dict)
        vec = np.zeros(num_go)
        for go in goset:
            vec[aspect_go2vec_dict[go]] = 1
        return vec
    
    vec = list()
    for go in goset:
        vec.append(aspect_go2vec_dict[go])
    vec = list(sorted(vec))
    return vec

def find_restratifi_term(protein2fold,aspect_train_term_matrix,min_fold_num=2):
    ''' Find a list of terms belonging to only one fold'''
    restratifi_term_list=[]
    n_protein, n_term = aspect_train_term_matrix.shape
    for term in range(n_term):
        unique_list=np.unique(protein2fold[aspect_train_term_matrix[:,term]>0])
        if len(unique_list)>=min_fold_num:
            continue
        restratifi_term_list.append(term)
        print(f"term {term} unique to fold {unique_list[0]}")
    return restratifi_term_list

def restratifi_term(protein2fold,protein2clust,restratifi_term_list,
    aspect_train_term_matrix, n_splits):
    n_protein, n_term = aspect_train_term_matrix.shape
    for t,term in enumerate(restratifi_term_list):
        aspect_train_term_list=aspect_train_term_matrix[:,term]
        restratifi_protein_list=[(idx,protein2clust[idx]) for idx in range(
            n_protein) if aspect_train_term_list[idx]]
        unique_list=np.unique([c for i,c in restratifi_protein_list])
        if len(unique_list)==1:
            print(f"all {len(restratifi_protein_list)} protein for term {term} belong to only one cluster")
            continue
            print(f"all {len(restratifi_protein_list)} protein for term {term} belong to only one cluster; evenly split it through all {n_splits} fold")
            for i,(idx,c) in enumerate(restratifi_protein_list):
                #protein2fold[idx]=(i)%n_splits
                protein2fold[idx]=(t+i)%n_splits
        else:
            print(f"{len(restratifi_protein_list)} protein for term {term} belong to {len(unique_list)} cluster; split it by cluster")
            clust2fold=dict()
            for c in unique_list:
                clust2fold[c]=(len(clust2fold))%n_splits
                #clust2fold[c]=(t+len(clust2fold))%n_splits
            for i,(idx,c) in enumerate(restratifi_protein_list):
                protein2fold[idx]=clust2fold[c]
    return protein2fold

def clust_split(seq_clust,    # list  
    aspect_train_seq_list,    # np.array
    aspect_train_term_matrix, # N * terms
    n_splits=5,               # number of folds
    stratifi=True             # ensure every fold has every label
    ):

    n_protein, n_term = aspect_train_term_matrix.shape
    print(f"split {n_protein} protein and {n_term} EC from {len(seq_clust)} cluster into {n_splits} fold")
    
    if stratifi:
        min_pos_labels=aspect_train_term_matrix.sum(axis=0).min()
        if min_pos_labels<n_splits:
            print(f"ERROR! cannot split {min_pos_labels} into {n_splits} folds")
            exit(1)

    protein2fold  = np.zeros(n_protein,dtype=int) - 1
    protein2clust = np.zeros(n_protein,dtype=int) -1
    seq2idx =dict()
    for idx,seq in enumerate(aspect_train_seq_list):
        seq2idx[seq]=idx
    for c,clust in enumerate(seq_clust):
        fold_n_list=[(sum(protein2fold==n),n) for n in range(n_splits)]
        fold_n_list.sort()
        #idx_list=[seq2idx[mem] for mem in clust if mem in seq2idx]
        #protein2fold[idx_list]= fold_n_list[0][1]
        idx_list=[]
        for mem in clust:
            if not mem in seq2idx:
                continue
            idx=seq2idx[mem]
            protein2clust[idx]=c
            protein2fold[idx] =fold_n_list[0][1]

    if stratifi:
        #for min_fold_num in range(n_splits,1,-1):
            #restratifi_term_list=find_restratifi_term(protein2fold,
                #aspect_train_term_matrix,min_fold_num)
            #print(f"{len(restratifi_term_list)} terms unique to <{min_fold_num} folds")
            #protein2fold=restratifi_term(protein2fold,protein2clust,
                #restratifi_term_list,aspect_train_term_matrix,n_splits)
        #restratifi_term_list=find_restratifi_term(protein2fold,
            #aspect_train_term_matrix,2)
        #print(f"{len(restratifi_term_list)} terms unique to one fold")

        restratifi_term_list=find_restratifi_term(protein2fold,
            aspect_train_term_matrix)
        if len(restratifi_term_list):
            print(f"{len(restratifi_term_list)} terms unique to one fold")
            protein2fold=restratifi_term(protein2fold,protein2clust,
                restratifi_term_list,aspect_train_term_matrix,n_splits)
            restratifi_term_list=find_restratifi_term(protein2fold,
                aspect_train_term_matrix)
            print(f"{len(restratifi_term_list)} terms unique to one fold")
            if len(restratifi_term_list):
                protein2fold=restratifi_term(protein2fold,protein2clust,
                    restratifi_term_list,aspect_train_term_matrix,n_splits)
                restratifi_term_list=find_restratifi_term(protein2fold,
                    aspect_train_term_matrix)
                print(f"{len(restratifi_term_list)} terms unique to one fold")

    folds=[]
    for n in range(n_splits):
        train_idx = [i for i in range(n_protein) if protein2fold[i]!=n]
        val_idx   = [i for i in range(n_protein) if protein2fold[i]==n]
        folds.append(( np.array(train_idx), np.array(val_idx)))
        print(f"fold {n}: {len(train_idx)} training instance; {len(val_idx)} validation instance")
    return folds # train_idx, val_idx   # np.array

def MultilabelStratifiedKFold_clust(
    seq_clust,                # list of sequence cluster
    aspect_train_seq_list,    # np.array
    aspect_train_term_matrix, # N * terms
    n_splits=5,               # number of folds
    random_state=1234567890,  # random seed
    ):

    n_protein, n_term = aspect_train_term_matrix.shape
    print(f"split {n_protein} protein and {n_term} EC into {n_splits} fold")

    seq2idx =dict()
    for idx,name in enumerate(aspect_train_seq_list):
        seq2idx[name]=idx
    
    clust2prot    = []
    for clust in seq_clust:
        idx_list = [seq2idx[name] for name in clust if name in seq2idx]
        if len(idx_list)==0:
            continue
        clust2prot.append(idx_list)
    n_clust = len(clust2prot)
    
    print(f"stratify {n_clust} out of {len(seq_clust)} cluster")
    aspect_clust_term_matrix = np.zeros((n_clust, n_term),dtype=int)
    aspect_clust_seq_list = np.arange(n_clust)
    for c,idx_list in enumerate(clust2prot):
        aspect_clust_term_matrix[c,:]=(
        aspect_train_term_matrix[idx_list,:].sum(axis=0)>0)

    kf = MultilabelStratifiedKFold(n_splits=n_splits, 
        random_state=random_state, shuffle=True)
    clust_folds = kf.split(aspect_clust_seq_list, aspect_clust_term_matrix)

    folds=[]
    for i, (train_clust_idx, val_clust_idx) in enumerate(clust_folds):
        train_idx = []
        val_idx   = []
        train_label = (aspect_train_term_matrix[train_clust_idx,:].sum(axis=0)>0)
        val_label   = (aspect_train_term_matrix[  val_clust_idx,:].sum(axis=0)>0)
        train_label_sum = sum(train_label)
        val_label_sum   = sum(val_label)
        overlap_label_sum = sum(train_label * val_label) 
        print(f"fold {i}: {train_label_sum} training labels; {val_label_sum} validation labels; {overlap_label_sum} overlap labels; {n_term} total labels")
        for c in train_clust_idx:
            train_idx += clust2prot[c]
        for c in val_clust_idx:
            val_idx   += clust2prot[c]
        folds.append(( np.array(train_idx), np.array(val_idx)))
        print(f"fold {i}: expand {len(train_clust_idx)} cluster into {len(train_idx)} training instance; expand {len(val_clust_idx)} cluster into {len(val_idx)} validation instance")
    return folds # train_idx, val_idx   # np.array

def main(train_terms_tsv:str, train_seqs_fasta:str, out_dir:str, 
    min_count_dict:dict=None, seed:int=1234567890):
    """
    Main function to prepare the data for training the model.

    Args:
        train_terms_tsv (str): path to the tsv file of the training terms
        train_seqs_fasta (str): path to the fasta file of the training sequences
        make_db (bool, optional): Whether to make the database. Defaults to True.
    """
    train_seq_dict = get_seq_dict(train_seqs_fasta)
    train_terms = pd.read_csv(train_terms_tsv, sep='\t')
    train_terms = prop_parents(train_terms)

    seq_clust=[]
    fp=open(train_seqs_fasta+".cdhit.clstr")
    for block in ('\n'+fp.read()).split('\n>'):
        if len(block.strip())==0:
            continue
        seq_clust.append([])
        for line in block.splitlines()[1:]:
            seq_clust[-1].append(
                line.split(', >')[1].split('.')[0])
    fp.close()

    if min_count_dict is not None:
        # filter out terms with less than min_count_dict
        ec1_terms_freq = train_terms[train_terms['aspect'] == 'EC1']['term'].value_counts()
        ec2_terms_freq = train_terms[train_terms['aspect'] == 'EC2']['term'].value_counts()
        ec3_terms_freq = train_terms[train_terms['aspect'] == 'EC3']['term'].value_counts()
        ec4_terms_freq = train_terms[train_terms['aspect'] == 'EC4']['term'].value_counts()
        selected_ec1_terms = set(ec1_terms_freq[ec1_terms_freq >= min_count_dict['EC1']].index)
        selected_ec2_terms = set(ec2_terms_freq[ec2_terms_freq >= min_count_dict['EC2']].index)
        selected_ec3_terms = set(ec3_terms_freq[ec3_terms_freq >= min_count_dict['EC3']].index)
        selected_ec4_terms = set(ec4_terms_freq[ec4_terms_freq >= min_count_dict['EC4']].index)
        selected_terms = selected_ec1_terms | selected_ec2_terms | selected_ec3_terms | selected_ec4_terms
        train_terms = train_terms[train_terms['term'].isin(selected_terms)]
        print(f'terms in EC1 aspect with terms frequency greater than {min_count_dict["EC1"]}: {len(selected_ec1_terms)}')
        print(f'terms in EC2 aspect with terms frequency greater than {min_count_dict["EC2"]}: {len(selected_ec2_terms)}')
        print(f'terms in EC3 aspect with terms frequency greater than {min_count_dict["EC3"]}: {len(selected_ec3_terms)}')
        print(f'terms in EC4 aspect with terms frequency greater than {min_count_dict["EC4"]}: {len(selected_ec4_terms)}\n')

    selected_terms_by_aspect = {
       'EC' : [],
       'EC1': [],
       'EC2': [],
       'EC3': [],
       'EC4': [],
    }

    get_aspect=dict()
    fp=open(train_terms_tsv)
    for line in fp.read().splitlines()[1:]:
        EntryID,aspect,term=line.split('\t')
        get_aspect[term]=aspect
    fp.close()
    for term in selected_terms:
        aspect = get_aspect[term]
        selected_terms_by_aspect[aspect].append(term)
        selected_terms_by_aspect['EC'].append(term)


    # topological sort the terms based on parent, child relationship
    #for k, v in selected_terms_by_aspect.items():
        #selected_terms_by_aspect[k] = oboTools.top_sort(v)[::-1]  
    
    # EntryIDs
    selected_entry_by_aspect = {
        'EC1': list(sorted(train_terms[train_terms['aspect'] == 'EC1']['EntryID'].unique())),
        'EC2': list(sorted(train_terms[train_terms['aspect'] == 'EC2']['EntryID'].unique())),
        'EC3': list(sorted(train_terms[train_terms['aspect'] == 'EC3']['EntryID'].unique())),
        'EC4': list(sorted(train_terms[train_terms['aspect'] == 'EC4']['EntryID'].unique())),
    }
    selected_entry_by_aspect['EC']  = list(sorted(train_terms['EntryID'].unique()))
    selected_terms_by_aspect['EC']  = sorted(selected_terms_by_aspect['EC1']
                                  ) + sorted(selected_terms_by_aspect['EC2']
                                  ) + sorted(selected_terms_by_aspect['EC3']
                                  ) + sorted(selected_terms_by_aspect['EC4'])


    # go2vec
    aspect_go2vec_dict = {
        'EC' : dict(),
        'EC1': dict(),
        'EC2': dict(),
        'EC3': dict(),
        'EC4': dict(),
    }
    for aspect in aspect_go2vec_dict:
        for i, term in enumerate(selected_terms_by_aspect[aspect]):
            aspect_go2vec_dict[aspect][term] = i

    aspect_train_term_grouped_dict = {
       'EC' : train_terms.reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC1': train_terms[train_terms['aspect'] == 'EC1'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC2': train_terms[train_terms['aspect'] == 'EC2'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC3': train_terms[train_terms['aspect'] == 'EC3'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC4': train_terms[train_terms['aspect'] == 'EC4'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    }

    for aspect in ['EC']:
        train_dir = os.path.join(out_dir, aspect)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        # write term_list
        aspect_term_list = selected_terms_by_aspect[aspect].copy()
        with open(os.path.join(train_dir, 'term_list.txt'), 'w') as f:
            for term in aspect_term_list:
                f.write(f'{term}\n')

        aspect_train_seq_list = selected_entry_by_aspect[aspect].copy()
        aspect_train_seq_list = np.array(aspect_train_seq_list)
        aspect_train_term_matrix = np.vstack([
            goset2vec(aspect_train_term_grouped_dict[aspect][entry], 
            aspect_go2vec_dict[aspect], fixed_len=True
            ) for entry in aspect_train_seq_list])
        print(f'{aspect} label dim: {aspect_train_term_matrix.shape}')

        #kf = MultilabelStratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        #folds = kf.split(aspect_train_seq_list, aspect_train_term_matrix)
        
        #kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        #folds = kf.split(aspect_train_seq_list)
        
        #folds = clust_split(seq_clust, aspect_train_seq_list, aspect_train_term_matrix, n_splits=5)
        
        folds = MultilabelStratifiedKFold_clust(seq_clust, aspect_train_seq_list,  
            aspect_train_term_matrix, n_splits=5, random_state=seed)

        for i, (train_idx, val_idx) in enumerate(folds):
            print(f'creating {aspect} fold {i}: {len(train_idx)} training; {len(val_idx)} validation')
            train_names = aspect_train_seq_list[train_idx]
            val_names = aspect_train_seq_list[val_idx]
            aspect_fold_train_label_npy = aspect_train_term_matrix[train_idx, :]
            aspect_fold_val_label_npy = aspect_train_term_matrix[val_idx, :]

            # convert to sparse matrix
            aspect_fold_train_label_npy = ssp.csr_matrix(aspect_fold_train_label_npy)
            aspect_fold_val_label_npy = ssp.csr_matrix(aspect_fold_val_label_npy)

            ssp.save_npz(os.path.join(train_dir, f'{aspect}_train_labels_fold{i}.npz'), aspect_fold_train_label_npy)
            ssp.save_npz(os.path.join(train_dir, f'{aspect}_valid_labels_fold{i}.npz'), aspect_fold_val_label_npy)

            np.save(os.path.join(train_dir, f'{aspect}_train_names_fold{i}.npy'), train_names)
            np.save(os.path.join(train_dir, f'{aspect}_valid_names_fold{i}.npy'), val_names)
        print(f'{aspect} done\n')            
    return

def split_db(db:str):
    out_dir = os.path.join(Data_dir, "network_training_data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_seq_dict = get_seq_dict(train_seqs_fasta)
    n_splits=5
    filename_list=[]
    for aspect in ['EC']:
        train_dir = os.path.join(out_dir, aspect)
        for i in range(n_splits):
            print(f'reading {aspect} train fold {i}')
            train_names=np.load(os.path.join(train_dir, 
                 f'{aspect}_train_names_fold{i}.npy'))
            txt=''.join(['>'+name+'\n'+train_seq_dict[name]+'\n' \
                    for name in train_names])
            train_filename=os.path.join(train_dir,f'{aspect}_train_names_fold{i}')
            fp=open(train_filename+".fasta",'w')
            fp.write(txt)
            fp.close()
            cmd=settings['mmseqs']+' createdb '+train_filename+'.fasta '+train_filename
            print(cmd)
            os.system(cmd)

            print(f'reading {aspect} valid fold {i}')
            val_names=np.load(os.path.join(train_dir, 
                 f'{aspect}_valid_names_fold{i}.npy'))
            txt=''.join(['>'+name+'\n'+train_seq_dict[name]+'\n' \
                    for name in val_names])
            val_filename=os.path.join(train_dir,f'{aspect}_val_names_fold{i}')
            fp=open(val_filename+".fasta",'w')
            fp.write(txt)
            fp.close()

            filename_list.append((train_filename,val_filename))
        print(f'{aspect} done\n')            
    return filename_list

def update_data():
    """
    Download the newest UNIPROT GOA file and preprocess to the csv and fasta
    """
    # only grep exp data
    rawdir=os.path.dirname(settings['train_ec_fasta'])
    if not os.path.isdir(rawdir):
        os.makedirs(rawdir)
    for filename in [settings['train_ec_tsv'],
                     settings['train_nonec_tsv'],
                     settings['train_ec_fasta'],
                     settings['train_nonec_fasta']]:
        if not os.path.isfile(filename):
            print("ERROR! no such file "+filename)
            exit(1)

    cmd=settings['mmseqs']+' createdb '+settings['train_ec_fasta']+' '+settings['db']
    print(cmd)
    os.system(cmd)
    if not os.path.isdir(settings['tmp_dir']):
        os.makedirs(settings['tmp_dir'])
    cmd=settings['mmseqs']+' createindex '+settings['db']+' '+settings['tmp_dir']
    print(cmd)
    os.system(cmd)

    print("read active site from "+settings['train_ec_tsv'])
    bs_dict=dict()
    fp=open(settings['train_ec_tsv'])
    for line in fp.read().splitlines():
        if line.startswith('#'):
            continue
        items=line.split('\t')
        target=items[0]
        binding=items[4].replace('..',',')
        act_site=items[5].replace('..',',')
        if not binding and not act_site:
            continue
        bs_list=[]
        if binding:
            bs_list+=[int(s) for s in binding.split(',')]
        if act_site:
            bs_list+=[int(s) for s in act_site.split(',')]
        if not target in bs_dict:
            bs_dict[target]=bs_list
        else:
            bs_dict[target]+=bs_list


    print("prepare "+settings['train_seqs1_fasta'])
    totalN=0
    seleN=0
    seq_list=[]
    target_list=[]
    fp=open(settings['train_ec_fasta'])
    for block in ('\n'+fp.read()).split('\n>'):
        if not block.strip():
            continue
        lines=block.splitlines()
        target=lines[0].split()[0]
        sequence=''.join(lines[1:])
        totalN+=1
        if len(sequence)<30:
            continue
        if len(sequence)>2048:
            if not target in bs_dict:
                print("ERROR! "+target+" too long (L=%d) and lack site"%len(sequence))
                continue
            min_bs=min(bs_dict[target])-1
            max_bs=max(bs_dict[target])-1
            if max_bs-min_bs>=2048:
                print("ERROR! "+target+" too long (L=%d), site=%d..%d"%(
                    len(sequence),min_bs,max_bs))
                continue
            mid_bs = int((max_bs + min_bs ) / 2)
            min_bs = mid_bs - 1024
            if min_bs<0:
                min_bs = 0
            max_bs = min_bs + 2048
            if max_bs >=len(sequence):
                min_bs = len(sequence)-2048
            print("WARNING! trim "+target+" from L=%d to L=2048"%(len(sequence)))
            sequence=sequence[min_bs:(min_bs+2048)]
        seleN+=1
        seq_list.append((len(sequence),'>'+target+'\n'+sequence+'\n'))
        target_list.append(target)
    fp.close()
    #seq_list.sort(reverse=True)
    
    print("writing %d out of %d sequence"%(seleN,totalN))
    txt=''.join([seq for L,seq in seq_list])
    fp=open(settings['train_seqs1_fasta'],'w')
    fp.write(txt)
    fp.close()
    
    print("prepare "+settings['train_seqs2_fasta'])
    fp=open(settings['train_nonec_fasta'])
    for block in ('\n'+fp.read()).split('\n>'):
        if not block.strip():
            continue
        lines=block.splitlines()
        target=lines[0].split()[0]
        sequence=''.join(lines[1:])
        totalN+=1
        if len(sequence)<30:
            continue
        if len(sequence)>2048:
            print("ERROR! "+target+" too long (L=%d)"%len(sequence))
            continue
        seleN+=1
        seq_list.append((len(sequence),'>'+target+'\n'+sequence+'\n'))
        target_list.append(target)
    fp.close()
    #seq_list.sort(reverse=True)
    #random.shuffle(seq_list)
    target_set=set(target_list)
    print("writing %d out of %d sequence"%(seleN,totalN))
    txt=''.join([seq for L,seq in seq_list])
    fp=open(settings['train_seqs2_fasta'],'w')
    fp.write(txt)
    fp.close()

    cmd_list=[
        settings['cdhit_path']+" -i "+settings['train_seqs1_fasta']+" -o "+settings['train_seqs1_fasta']+".cdhit -c 0.6 -M 5000 -n 4 -T 16 -g 1 -sc 1",
        settings['cdhit_path']+" -i "+settings['train_seqs2_fasta']+" -o "+settings['train_seqs2_fasta']+".cdhit -c 0.6 -M 5000 -n 4 -T 16 -g 1 -sc 1"]
    for cmd in cmd_list:
        print(cmd)
        os.system(cmd)
    
    print("prepare "+settings['train_terms1_tsv'])
    line_list=[]
    train_ec_dict=dict()
    fp=open(settings['train_ec_tsv'])
    for line in fp.read().splitlines():
        if line.startswith('#'):
            continue
        target,ECnumber=line.split('\t')[:2]
        if not target in target_set:
            continue
        EC_list=ECnumber.split('.')
        for i in range(1,5):
            if EC_list[i-1]=='-':
                continue
            ECnumber='.'.join(EC_list[:i]+['-']*(4-i))
            line_list.append("%s\tEC%d\t%s\n"%(target,i,ECnumber))
    fp.close()
    txt='EntryID\taspect\tterm\n'+''.join(list(set(line_list)))
    fp=open(settings['train_terms1_tsv'],'w')
    fp.write(txt)
    fp.close()
    
    print("prepare "+settings['train_terms2_tsv'])
    nonec_list=[]
    fp=open(settings['train_nonec_tsv'])
    for line in fp.read().splitlines():
        if line.startswith('#'):
            continue
        target=line.split('\t')[0]
        if not target in target_set:
            continue
        nonec_list.append(target)
    fp.close()
    for target in set(nonec_list):
        line_list.append(target+"\tEC1\t0.-.-.-\n")
    txt='EntryID\taspect\tterm\n'+''.join(list(set(line_list)))
    fp=open(settings['train_terms2_tsv'],'w')
    fp.write(txt)
    fp.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--Data_dir', type=str, default=settings['DATA_DIR'], help='path to the directory of the data')
    parser.add_argument('--min_ec',  type=int, default=10, help='minimum number instances of EC number')
    parser.add_argument('--seed', type=int, default=1234567890)
    args = parser.parse_args()
    args.Data_dir = os.path.abspath(args.Data_dir)
    min_count_dict = {
        'EC' : args.min_ec,
        'EC1': args.min_ec,
        'EC2': args.min_ec,
        'EC3': args.min_ec,
        'EC4': args.min_ec,
    }
    seed = args.seed

    Data_dir = args.Data_dir
    for folder in [settings['TRAIN_DATA_CLEAN_DIR1'],
                   settings['TRAIN_DATA_CLEAN_DIR2']]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    update_data()

    for idx in ['1','2']:
        IC(settings['train_terms'+idx+'_tsv'], settings['ia_file'+idx])
        main(settings['train_terms'+idx+'_tsv'], 
            settings['train_seqs'+idx+'_fasta'], 
            settings['TRAIN_DATA_CLEAN_DIR'+idx],
            min_count_dict, seed)
    
    # extract embeddings
    print('Extracting embeddings...')
    extract_homolog_embeddings(settings['train_seqs2_fasta'], settings['db'])
    #extract_embeddings(settings['train_seqs2_fasta'])
