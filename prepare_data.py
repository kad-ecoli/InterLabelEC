import pandas as pd
import os, sys, subprocess, argparse
import numpy as np
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import shutil

from plm import PlmEmbed
from settings import settings_dict as settings

docstring = """
example usage:
    python prepare_data.py -train Data/train_raw/train_terms.tsv -train_seqs Data/train_raw/train_seq.fasta -d Data --make_db --ia

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


def make_db(Data_dir:str, terms_df:pd.DataFrame, seq_dict:dict):
    DatabaseDir = settings['alignment_db']
    goa_dir = settings['alignment_labels']

    if not os.path.exists(DatabaseDir):
        os.makedirs(DatabaseDir)
    if not os.path.exists(goa_dir):
        os.makedirs(goa_dir)

    for aspect in ['EC1', 'EC2', 'EC3', 'EC4']:
        aspect_db_dir = os.path.join(DatabaseDir, aspect)
        if not os.path.exists(aspect_db_dir):
            os.makedirs(aspect_db_dir)
        cur_aspect_df = terms_df[terms_df['aspect'] == aspect].copy()
        # group by EntryID, apply set to term
        cur_aspect_df = cur_aspect_df.groupby(['EntryID'])['term'].apply(set).reset_index()
        # backprop parent
        #cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: oboTools.backprop_terms(x))
        cur_aspect_df['term'] = cur_aspect_df['term'].apply(lambda x: ','.join(x))
        unique_entryids = set(cur_aspect_df['EntryID'].unique())
        if len(unique_entryids) == 0:
            raise Exception(f"Error: no {aspect} terms found in dataframe, please make sure aspect is one of EC1, EC2, EC3, EC4")        
        # write to file
        label_path = os.path.join(goa_dir, f'{aspect}_Term')
        cur_aspect_df.to_csv(label_path, sep='\t', index=False, header=False)

        aspect_fasta_path = os.path.join(aspect_db_dir, 'AlignmentKNN.fasta')
        with open(aspect_fasta_path, 'w') as f:
            for entryid in unique_entryids:
                if entryid in seq_dict:
                    f.write(f">{entryid}\n{seq_dict[entryid]}\n")
                else:
                    print(f"Warning: {entryid} not found in seq_dict, skip it", file=sys.stderr)
        
        # create and diamond_db
        diamond_db_cmd = f'{settings["diamond_path"]} makedb --in {aspect_fasta_path} -d {os.path.join(aspect_db_dir, "AlignmentKNN")}'
        print('creating diamond_db...')
        subprocess.run(diamond_db_cmd, shell=True, check=True)       

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

def perpare_test(test_terms_tsv:str, test_seqs_fasta:str, Data_dir:str, selected_terms_by_aspect:dict):
    """
    prepare the test data, this is mainly for the fine-tuning of precision and recall
    estimation of the model.

    Args:
        test_terms_tsv (str): path to the tsv file of the test terms
        test_seqs_fasta (str): path to the fasta file of the test sequences
        Data_dir (str): path to the directory of the data
        selected_terms_by_aspect (dict): dict of selected terms, key is aspect, value is a set of terms
    """
    test_seq_dict = get_seq_dict(test_seqs_fasta)
    present_seqs = set(test_seq_dict.keys())
    test_terms = pd.read_csv(test_terms_tsv, sep='\t')
    test_terms = prop_parents(test_terms)
    all_terms = set(selected_terms_by_aspect['EC1']
            ) | set(selected_terms_by_aspect['EC2']
            ) | set(selected_terms_by_aspect['EC3']
            ) | set(selected_terms_by_aspect['EC4'])
    test_terms = test_terms[test_terms['term'].isin(all_terms)]
    test_terms = test_terms[test_terms['EntryID'].isin(present_seqs)]

    selected_entry_by_aspect = {
        'EC1': list(sorted(test_terms[test_terms['aspect'] == 'EC1']['EntryID'].unique())),
        'EC2': list(sorted(test_terms[test_terms['aspect'] == 'EC2']['EntryID'].unique())),
        'EC3': list(sorted(test_terms[test_terms['aspect'] == 'EC3']['EntryID'].unique())),
        'EC4': list(sorted(test_terms[test_terms['aspect'] == 'EC4']['EntryID'].unique())),
    }
    print(f'number of proteins in EC1 aspect: {len(selected_entry_by_aspect["EC1"])}')
    print(f'number of proteins in EC2 aspect: {len(selected_entry_by_aspect["EC2"])}')
    print(f'number of proteins in EC3 aspect: {len(selected_entry_by_aspect["EC3"])}')
    print(f'number of proteins in EC4 aspect: {len(selected_entry_by_aspect["EC4"])}')

    # go2vec
    aspect_go2vec_dict = {
        'EC1': dict(),
        'EC2': dict(),
        'EC3': dict(),
        'EC4': dict(),
    }
    for aspect in ['EC1', 'EC2', 'EC3', 'EC4']:
        for i, term in enumerate(selected_terms_by_aspect[aspect]):
            aspect_go2vec_dict[aspect][term] = i
    
    aspect_test_term_grouped_dict = {
        'EC1': test_terms[test_terms['aspect'] == 'EC1'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
        'EC2': test_terms[test_terms['aspect'] == 'EC2'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
        'EC3': test_terms[test_terms['aspect'] == 'EC3'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
        'EC4': test_terms[test_terms['aspect'] == 'EC4'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    }

    out_dir = settings["TRAIN_DATA_CLEAN_DIR"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for aspect in ['EC1', 'EC2', 'EC3', 'EC4']:
        test_dir = os.path.join(out_dir, aspect)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        aspect_test_seq_list = selected_entry_by_aspect[aspect].copy()
        aspect_test_seq_list = np.array(aspect_test_seq_list)
        aspect_test_term_matrix = np.vstack([goset2vec(aspect_test_term_grouped_dict[aspect][entry], 
            aspect_go2vec_dict[aspect], fixed_len=True) for entry in aspect_test_seq_list])
        print(f'{aspect} label dim: {aspect_test_term_matrix.shape}')

        # convert to sparse matrix
        aspect_test_term_matrix = ssp.csr_matrix(aspect_test_term_matrix)

        np.save(os.path.join(test_dir, f'{aspect}_test_names.npy'), aspect_test_seq_list)
        ssp.save_npz(os.path.join(test_dir, f'{aspect}_test_labels.npz'), aspect_test_term_matrix)

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

def main(train_terms_tsv:str, train_seqs_fasta:str, Data_dir:str, \
    make_alignment_db:bool=True, min_count_dict:dict=None, seed:int=1234567890, stratifi:bool=False,
    test_terms_tsv:str=None, test_seqs_fasta:str=None):
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

    if make_alignment_db:
        make_db(Data_dir, train_terms, train_seq_dict)
        print('database created\n')
    
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
       'EC1': set(),
       'EC2': set(),
       'EC3': set(),
       'EC4': set(),
    }

    get_aspect=dict()
    fp=open(train_terms_tsv)
    for line in fp.read().splitlines()[1:]:
        EntryID,aspect,term=line.split('\t')
        get_aspect[term]=aspect
    fp.close()
    for term in selected_terms:
        aspect = get_aspect[term]
        selected_terms_by_aspect[aspect].add(term)

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


    # go2vec
    aspect_go2vec_dict = {
        'EC1': dict(),
        'EC2': dict(),
        'EC3': dict(),
        'EC4': dict(),
    }
    for aspect in ['EC1', 'EC2', 'EC3', 'EC4']:
        for i, term in enumerate(selected_terms_by_aspect[aspect]):
            aspect_go2vec_dict[aspect][term] = i

    aspect_train_term_grouped_dict = {
       'EC1': train_terms[train_terms['aspect'] == 'EC1'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC2': train_terms[train_terms['aspect'] == 'EC2'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC3': train_terms[train_terms['aspect'] == 'EC3'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
       'EC4': train_terms[train_terms['aspect'] == 'EC4'].reset_index(drop=True).groupby('EntryID')['term'].apply(set),
    }

    out_dir = os.path.join(Data_dir, "network_training_data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if test_terms_tsv is not None and test_seqs_fasta is not None:
        perpare_test(test_terms_tsv, test_seqs_fasta, Data_dir, selected_terms_by_aspect)

    for aspect in ['EC1', 'EC2', 'EC3', 'EC4']:
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
        
        folds = clust_split(seq_clust, aspect_train_seq_list, aspect_train_term_matrix, n_splits=5)

        for i, (train_idx, val_idx) in enumerate(folds):
            print(f'creating {aspect} fold {i}...')
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


def extract_embeddings(train_seq_fasta:str):
    """extract esm embeddings from the training sequences fasta file

    Args:
        train_seq_fasta (str): path to the fasta file of the training sequences
    """
    plm = PlmEmbed(
        fasta_file=train_seq_fasta,
        working_dir=settings['tmp_dir'],
        cache_dir=settings['embedding_dir'],
        model_path=settings['esm3b_path'],
    )
    plm.extract(
        include=['mean'],
    )
    # remove tmp dir
    if os.path.exists(settings['tmp_dir']):
        shutil.rmtree(settings['tmp_dir'], ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train","--train_terms_tsv", type=str, help="path to the tsv file of the training terms", default=None)
    parser.add_argument("-train_seqs","--train_seqs_fasta", type=str, help="path to the fasta file of the training sequences", default=None)
    parser.add_argument("-test", "--test_terms_tsv", type=str, help="path to the tsv file of the test terms", default=None)
    parser.add_argument("-test_seqs", "--test_seqs_fasta", type=str, help="path to the fasta file of the test sequences", default=None)
    parser.add_argument('-d', '--Data_dir', type=str, default=settings['DATA_DIR'], help='path to the directory of the data')
    parser.add_argument('--make_db', action='store_true', help='whether to make the alignment database')
    parser.add_argument('--min_ec',  type=int, default=10, help='minimum number instances of EC number')
    #parser.add_argument('--min_ec1', type=int, default=10, help='minimum number instances of EC number (digit 1)')
    #parser.add_argument('--min_ec2', type=int, default=10, help='minimum number instances of EC number (digit 2)')
    #parser.add_argument('--min_ec3', type=int, default=10, help='minimum number instances of EC number (digit 3)')
    #parser.add_argument('--min_ec4', type=int, default=10, help='minimum number instances of EC number (digit 4)')
    parser.add_argument('--seed', type=int, default=1234567890)
    parser.add_argument('--ia', action='store_true', help='whether to calculate ia, the iA.txt would store under the Data_dir')
    parser.add_argument('--stratifi', action='store_true', help='whether to use stratified multi-label in kfold')
    args = parser.parse_args()
    args.Data_dir = os.path.abspath(args.Data_dir)
    min_count_dict = {
        'EC1': args.min_ec,
        'EC2': args.min_ec,
        'EC3': args.min_ec,
        'EC4': args.min_ec,
    }
    seed = args.seed
    make_alignment_db = args.make_db

    if args.train_terms_tsv is not None:
        args.train_terms_tsv = os.path.abspath(args.train_terms_tsv)
    elif os.path.exists(settings['train_terms_tsv']):
        args.train_terms_tsv = settings['train_terms_tsv']
    else:
        raise Exception(f'Error: train_terms_tsv not found at {args.train_terms_tsv}, please specify the path to the training terms tsv file')
    train_terms_tsv = args.train_terms_tsv

    if args.train_seqs_fasta is not None:
        args.train_seqs_fasta = os.path.abspath(args.train_seqs_fasta)
    elif os.path.exists(settings['train_seqs_fasta']):
        args.train_seqs_fasta = settings['train_seqs_fasta']
    else:
        raise Exception(f'Error: train_seqs_fasta not found at {args.train_seqs_fasta}, please specify the path to the training sequences fasta file')
    train_seqs_fasta = args.train_seqs_fasta

    if args.test_terms_tsv is not None:
        args.test_terms_tsv = os.path.abspath(args.test_terms_tsv)
    test_terms_tsv = args.test_terms_tsv
    if args.test_seqs_fasta is not None:
        args.test_seqs_fasta = os.path.abspath(args.test_seqs_fasta)
    test_seqs_fasta = args.test_seqs_fasta

    Data_dir = args.Data_dir
    stratifi = args.stratifi
    main(train_terms_tsv, train_seqs_fasta, Data_dir, make_alignment_db, min_count_dict, seed, stratifi, test_terms_tsv, test_seqs_fasta)

    if args.ia:
        ia_file = settings['ia_file']
        ia_script = settings['ia_script']
        cmd = f"python {ia_script} {train_terms_tsv} {ia_file}"
        print('Creating IA.txt...')
        subprocess.run(cmd, shell=True, check=True)
    
    # extract embeddings
    print('Extracting embeddings...')
    extract_embeddings(train_seqs_fasta)
    if test_seqs_fasta is not None:
        extract_embeddings(test_seqs_fasta)
