#!/usr/bin/env python3
import os, argparse
import torch
#from Bio import SeqIO
import numpy as np
import pickle
import scipy.sparse as ssp
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
import math #, json

from model import InterlabelGODataset, InterLabelResNet, Predictor
#from Network.model import InterlabelGODataset, InterLabelResNet
#from Network.model_utils import Predictor
#from utils.obo_tools import ObOTools
#from plm import PlmEmbed
from fasta2plm import fasta2esm,homolog2esm
from settings import settings_dict as settings
from settings import training_config

blosum62={
    'A':{'A': 4,'R':-1,'N':-2,'D':-2,'C': 0,'Q':-1,'E':-1,'G': 0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 0,'W':-3,'Y':-2,'V': 0,'B':-2,'Z':-1,'X': 0,'*':-4 },
    'R':{'A':-1,'R': 5,'N': 0,'D':-2,'C':-3,'Q': 1,'E': 0,'G':-2,'H': 0,'I':-3,'L':-2,'K': 2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3,'B':-1,'Z': 0,'X':-1,'*':-4 },
    'N':{'A':-2,'R': 0,'N': 6,'D': 1,'C':-3,'Q': 0,'E': 0,'G': 0,'H': 1,'I':-3,'L':-3,'K': 0,'M':-2,'F':-3,'P':-2,'S': 1,'T': 0,'W':-4,'Y':-2,'V':-3,'B': 3,'Z': 0,'X':-1,'*':-4 },
    'D':{'A':-2,'R':-2,'N': 1,'D': 6,'C':-3,'Q': 0,'E': 2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S': 0,'T':-1,'W':-4,'Y':-3,'V':-3,'B': 4,'Z': 1,'X':-1,'*':-4 },
    'C':{'A': 0,'R':-3,'N':-3,'D':-3,'C': 9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1,'B':-3,'Z':-3,'X':-2,'*':-4 },
    'Q':{'A':-1,'R': 1,'N': 0,'D': 0,'C':-3,'Q': 5,'E': 2,'G':-2,'H': 0,'I':-3,'L':-2,'K': 1,'M': 0,'F':-3,'P':-1,'S': 0,'T':-1,'W':-2,'Y':-1,'V':-2,'B': 0,'Z': 3,'X':-1,'*':-4 },
    'E':{'A':-1,'R': 0,'N': 0,'D': 2,'C':-4,'Q': 2,'E': 5,'G':-2,'H': 0,'I':-3,'L':-3,'K': 1,'M':-2,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2,'B': 1,'Z': 4,'X':-1,'*':-4 },
    'G':{'A': 0,'R':-2,'N': 0,'D':-1,'C':-3,'Q':-2,'E':-2,'G': 6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S': 0,'T':-2,'W':-2,'Y':-3,'V':-3,'B':-1,'Z':-2,'X':-1,'*':-4 },
    'H':{'A':-2,'R': 0,'N': 1,'D':-1,'C':-3,'Q': 0,'E': 0,'G':-2,'H': 8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y': 2,'V':-3,'B': 0,'Z': 0,'X':-1,'*':-4 },
    'I':{'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I': 4,'L': 2,'K':-3,'M': 1,'F': 0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V': 3,'B':-3,'Z':-3,'X':-1,'*':-4 },
    'L':{'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I': 2,'L': 4,'K':-2,'M': 2,'F': 0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V': 1,'B':-4,'Z':-3,'X':-1,'*':-4 },
    'K':{'A':-1,'R': 2,'N': 0,'D':-1,'C':-3,'Q': 1,'E': 1,'G':-2,'H':-1,'I':-3,'L':-2,'K': 5,'M':-1,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2,'B': 0,'Z': 1,'X':-1,'*':-4 },
    'M':{'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q': 0,'E':-2,'G':-3,'H':-2,'I': 1,'L': 2,'K':-1,'M': 5,'F': 0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V': 1,'B':-3,'Z':-1,'X':-1,'*':-4 },
    'F':{'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I': 0,'L': 0,'K':-3,'M': 0,'F': 6,'P':-4,'S':-2,'T':-2,'W': 1,'Y': 3,'V':-1,'B':-3,'Z':-3,'X':-1,'*':-4 },
    'P':{'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P': 7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2,'B':-2,'Z':-1,'X':-2,'*':-4 },
    'S':{'A': 1,'R':-1,'N': 1,'D': 0,'C':-1,'Q': 0,'E': 0,'G': 0,'H':-1,'I':-2,'L':-2,'K': 0,'M':-1,'F':-2,'P':-1,'S': 4,'T': 1,'W':-3,'Y':-2,'V':-2,'B': 0,'Z': 0,'X': 0,'*':-4 },
    'T':{'A': 0,'R':-1,'N': 0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S': 1,'T': 5,'W':-2,'Y':-2,'V': 0,'B':-1,'Z':-1,'X': 0,'*':-4 },
    'W':{'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F': 1,'P':-4,'S':-3,'T':-2,'W':11,'Y': 2,'V':-3,'B':-4,'Z':-3,'X':-2,'*':-4 },
    'Y':{'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H': 2,'I':-1,'L':-1,'K':-2,'M':-1,'F': 3,'P':-3,'S':-2,'T':-2,'W': 2,'Y': 7,'V':-1,'B':-3,'Z':-2,'X':-1,'*':-4 },
    'V':{'A': 0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I': 3,'L': 1,'K':-2,'M': 1,'F':-1,'P':-2,'S':-2,'T': 0,'W':-3,'Y':-1,'V': 4,'B':-3,'Z':-2,'X':-1,'*':-4 },
    'B':{'A':-2,'R':-1,'N': 3,'D': 4,'C':-3,'Q': 0,'E': 1,'G':-1,'H': 0,'I':-3,'L':-4,'K': 0,'M':-3,'F':-3,'P':-2,'S': 0,'T':-1,'W':-4,'Y':-3,'V':-3,'B': 4,'Z': 1,'X':-1,'*':-4 },
    'Z':{'A':-1,'R': 0,'N': 0,'D': 1,'C':-3,'Q': 3,'E': 4,'G':-2,'H': 0,'I':-3,'L':-3,'K': 1,'M':-1,'F':-3,'P':-1,'S': 0,'T':-1,'W':-3,'Y':-2,'V':-2,'B': 1,'Z': 4,'X':-1,'*':-4 },
    'X':{'A': 0,'R':-1,'N':-1,'D':-1,'C':-2,'Q':-1,'E':-1,'G':-1,'H':-1,'I':-1,'L':-1,'K':-1,'M':-1,'F':-1,'P':-2,'S': 0,'T': 0,'W':-2,'Y':-1,'V':-1,'B':-1,'Z':-1,'X': 1,'*':-4 },
    '*':{'A':-4,'R':-4,'N':-4,'D':-4,'C':-4,'Q':-4,'E':-4,'G':-4,'H':-4,'I':-4,'L':-4,'K':-4,'M':-4,'F':-4,'P':-4,'S':-4,'T':-4,'W':-4,'Y':-4,'V':-4,'B':-4,'Z':-4,'X':-4,'*': 1 },
}

def System(cmd):
    print(cmd)
    os.system(cmd)
    return

class InterLabelGO_pipeline:
    def __init__(self,
        working_dir:str,
        fasta_file:str,
        pred_batch_size:int=512,
        device:str='cuda',
        top_terms:int=500, # number of top terms to be keeped in the prediction
        aspects:list=['EC'], # aspects to predict
        ## the following parameters should be fixed if you want to use the pretrained model
        repr_layers:list=training_config['repr_layers'],
        embed_batch_size:int=4096, # note this might take around 15GB of vram, if you don't have enough vram, you can set this to 2048
        embed_model_name:str="esmc_600m_2024_12_v0",
        embed_model_path:str=settings['esm3b_path'],
        include:list=['mean'],
        model_dir:str=settings['MODEL_CHECKPOINT_DIR1'],
        result_file:str='',
    ) -> None:
        self.working_dir = os.path.abspath(working_dir)
        self.fasta_file = os.path.abspath(fasta_file)
        self.pred_batch_size = pred_batch_size
        
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.top_terms = top_terms
        self.aspects = aspects

        self.repr_layers = repr_layers
        self.embed_model_name = embed_model_name
        self.embed_model_path = embed_model_path
        self.include = include
        self.embed_batch_size = embed_batch_size

        self.model_dir = os.path.abspath(model_dir)
        self.result_file = os.path.join(self.working_dir, 'DL.tsv')
        if result_file:
            self.result_file=result_file


    def parse_fasta(self, fasta_file=None)->dict:
        '''
        parse fasta file

        args:
            fasta_file: fasta file path
        return:
            fasta_dict: fasta dictionary {id: sequence}
        '''
        if fasta_file is None:
            fasta_file = self.fasta_file

        #fasta_dict = {}
        #for record in SeqIO.parse(fasta_file, 'fasta'):
            #fasta_dict[record.id] = str(record.seq)
        
        fasta_dict=dict()
        fp=open(fasta_file)
        for block in ('\n'+fp.read()).split('\n>'):
            if len(block.strip())==0:
                continue
            lines=block.splitlines()
            header=lines[0]
            sequence=''.join(lines[1:])
            fasta_dict[header]=sequence
        fp.close()
        return fasta_dict  
    
    def get_embed_features(self):
        print("Extracting embeding features")
        feature_dir = os.path.join(self.working_dir, "embed_feature")
        #fasta2esm(self.fasta_file, feature_dir)
        homolog2esm(self.fasta_file, feature_dir, settings['db'])
        return feature_dir
    
    def create_name_npy(self):
        fasta_dict = self.parse_fasta(self.fasta_file)
        name_npy_path = os.path.join(self.working_dir, 'names.npy')
        names = np.array(list(fasta_dict.keys()))
        np.save(name_npy_path, names)
        return name_npy_path

    def predict(self, feature_dir:str):
        name_npy_path = self.create_name_npy() # create working_dir/names.npy file for DataLoader
        predictor = Predictor(
            model=None,
            PredictLoader=None,
            device=self.device,
            child_matrix=None,
            back_prop=True,
        )
        PredictDataset = InterlabelGODataset(
        features_dir=feature_dir,
        names_npy=name_npy_path,
        repr_layers=self.repr_layers,
        labels_npy=None,# set to because we are doing inference
        )
        PredictLoader = DataLoader(PredictDataset, batch_size=self.pred_batch_size, shuffle=False, num_workers=0)
        predictor.update_loader(PredictLoader)

        result_file = self.result_file
        #columns = ['EntryID','term','score','aspect', 'go_term_name']
        columns = ['EntryID','term','score']
        with open(result_file, 'w') as f:
            f.write('\t'.join(columns))
            f.write('\n')
        seeds_dict = dict()
        for aspect in self.aspects:
            aspect_model_dir = os.path.join(self.model_dir,aspect)
            if not os.path.exists(aspect_model_dir):
                print(f'No model found for {aspect} at {aspect_model_dir}')
                continue
            models = os.listdir(aspect_model_dir)
            models = [os.path.join(aspect_model_dir, model) for model in models if model.endswith('.pt')]

            if len(models) == 0:
                print(f"No model found in {aspect_model_dir}")
                continue
            child_matrix = ssp.load_npz(os.path.join(aspect_model_dir, 'child_matrix_ssp.npz')).toarray()
            predictions = []
            names = None
            print(f"Predicting {aspect}")

            # # only use one model for now
            # models = models[:1]

            for model_path in tqdm(models, desc=f'generate {aspect} prediction', ascii=' >='):
                model = InterLabelResNet.load_config(model_path)
                seed = model.seed
                if aspect not in seeds_dict:
                    seeds_dict[aspect] = {}
                seeds_dict[aspect][model_path] = seed
                # # save model again
                # model.save_config(model_path)
                model = model.to(self.device)
                predictor.update_model(model,child_matrix)
                predictor.back_prop = False
                protein_ids, y_preds = predictor.predict()
                predictions.append(y_preds)
                if names is None:
                    names = protein_ids

            predictions = sum(predictions)/len(predictions)

            term_list = model.go_term_list

            # convert to dataframe
            df = pd.DataFrame(predictions, columns=term_list, index=names)
            df = df.stack().reset_index()
            df.columns = columns
            df = df[df['score'] > 0.001]
            # sort by name then score
            df = df.sort_values(by=['EntryID','score'], ascending=[True, False])
            df = df.sort_values(by=['EntryID','score'], ascending=[True, False])
            df['aspect'] = aspect

            # only keep the top 500 terms
            df = df.groupby(['EntryID', 'aspect']).head(self.top_terms)
            df = df[columns]
            #df['go_term_name'] = df['term'].apply(lambda x: oboTools.goID2name(x))
            # write to tsv file
            df.to_csv(result_file, index=False, sep='\t', mode='a', header=False)

            System(settings["parse_isa"]+' '+result_file+' '+result_file)
            #print(seeds_dict)
        return

    def main(self):
        feature_dir = self.get_embed_features()
        self.predict(feature_dir)
        return

class combine_pipeline:
    def __init__(self,
        working_dir:str,
        fasta_file:str,
        pdb_dir:str='',
        result_file:str='',
    ) -> None:
        self.working_dir = os.path.abspath(working_dir)
        self.fasta_file = os.path.abspath(fasta_file)
        self.result_file = os.path.join(self.working_dir, 'combine.tsv')
        self.pdb_dir = ''
        if pdb_dir:
            self.pdb_dir = pdb_dir
        if result_file:
            self.result_file=result_file
    
    def read_annotation(self,infile_list=[]):
        sacc_list=[]
        for infile in infile_list:
            fp=open(infile)
            for line in fp.read().splitlines():
                items=line.split('\t')
                sacc_list.append(items[2])
            fp.close()
        sacc_set=set(sacc_list)
        read_sacc_set=len(sacc_set)
        print("%d templates"%len(sacc_set))
    
        exp_dict=dict()
        site_dict=dict()

        ECnumber="0.-.-.-"
        fp=open(settings["train_nonec_tsv"])
        for line in fp.read().splitlines():
            if line.startswith('#'):
                continue
            items=line.split('\t')
            sacc=items[0]
            if read_sacc_set and not sacc in sacc_set:
                continue
            exp_dict[sacc]=[ECnumber]
            if len(exp_dict) % 1000 ==0:
                print("parsed %d / %d templates"%(len(exp_dict),len(sacc_set)))
        fp.close()

        fp=open(settings["train_ec_tsv"])
        for line in fp.read().splitlines():
            if line.startswith('#'):
                continue
            items=line.split('\t')
            sacc=items[0]
            if read_sacc_set and not sacc in sacc_set:
                continue
            ECnumber=items[1]
            site_list=[]
            for col in [-2,-1]:
                if len(items[col])==0:
                    continue
                for site in items[col].split(','):
                    if not site:
                        continue
                    elif '..' in site:
                        sstart,send=site.split('..')
                        for s in range(int(sstart),int(send)+1):
                            site_list.append(s)
                    else:
                        site_list.append(int(site))
            site_list=set(site_list)
            if not sacc in exp_dict:
                exp_dict[sacc]=[]
            if len(site_list):
                if not sacc in site_dict:
                    site_dict[sacc]=site_list
                else:
                    site_dict[sacc].union(site_list)
            for e in range(4-ECnumber.count('-'),0,-1):
                ECnumber='.'.join(ECnumber.split('.')[:e]+['-']*(4-e))
                if not ECnumber in exp_dict[sacc]:
                    exp_dict[sacc].append(ECnumber)
            if len(exp_dict) % 1000 ==0:
                print("parsed %d / %d templates"%(len(exp_dict),len(sacc_set)))
        fp.close()
        return exp_dict,site_dict

    def parse_blast(self,infile,site_dict,infmt='blast'):
        similarity_dict=dict()
        for aa1 in blosum62:
            similarity_dict[aa1]=dict()
            for aa2 in blosum62[aa1]:
                similarity_dict[aa1][aa2]=2.*blosum62[aa1][aa2]/(
                      blosum62[aa1][aa1]+blosum62[aa2][aa2])
                if similarity_dict[aa1][aa2]<0:
                    similarity_dict[aa1][aa2]=0
    
        blast_dict=dict()
        fp=open(infile)
        stdout=fp.read()
        fp.close()
        for line in stdout.splitlines():
            if infmt=='diamond' or infmt=='blast':
                qacc,qlen,sacc,slen,evalue,bitscore,length,nident,qseq,qstart,sseq,sstart=line.split('\t')[:12]
            elif infmt=='foldseek':
                qacc,qlen,sacc,slen,evalue,bitscore,length,nident,qseq,qstart,sseq,sstart,qtmscore,stmscore=line.split('\t')[:14]
                if qacc in blast_dict and len(blast_dict[qacc])>=5:
                    continue
                qtmscore=float(qtmscore)
                stmscore=float(stmscore)
                TM=min((qtmscore,stmscore))
            else:
                print("ERROR! infmt=",infmt)
                exit()
            qlen=float(qlen)
            slen=float(slen)
            bitscore=float(bitscore)
            nident=float(nident)
            length=float(length)
            evalue=float(evalue)
            if nident<=1:
                nident*=length
            if not qacc in blast_dict:
                if len(blast_dict) % 1000 == 0:
                    print("parse %d %s"%(len(blast_dict),qacc))
                blast_dict[qacc]=[]
            ID=nident/max((qlen,slen))
            score=bitscore*ID
            if infmt=='foldseek':
                score=bitscore*ID*TM
            blast_dict[qacc].append([sacc, score, qseq, int(qstart), sseq, int(sstart), ID])

        count=0
        for qacc in blast_dict: 
            if count %1000 ==0:
                print("calculate local score %d %s"%(count,qacc))
            count+=1
            qacc_site_dict=dict()
            denominator=sum([blast_dict[qacc][i][1] for i in range(len(blast_dict[qacc]))])
            for items in blast_dict[qacc]:
                sacc=items[0]
                if not sacc in site_dict:
                    continue
                score =items[1]/denominator
                qseq  =items[2]
                qstart=items[3]
                sseq  =items[4]
                sstart=items[5]
                ID    =items[6]
                r1=qstart-1
                r2=sstart-1
                for p in range(len(qseq)):
                    aa1=qseq[p]
                    aa2=sseq[p]
                    r1+=aa1!='-'
                    r2+=aa2!='-'
                    if aa1!='-' and aa2!='-':
                        issite=(r2 in site_dict[sacc])
                        for ri in site_dict[sacc]:
                            if abs(ri-r2)<=2:
                                issite+=1
                        if issite:
                            if not r1 in qacc_site_dict:
                                qacc_site_dict[r1]=score*issite
                            else:
                                qacc_site_dict[r1]+=score*issite
            for i,items in enumerate(blast_dict[qacc]):
                sacc =items[0]
                score=items[1]
                if len(qacc_site_dict)>0:
                    qseq  =items[2]
                    qstart=items[3]
                    sseq  =items[4]
                    sstart=items[5]
                    r1=qstart-1
                    r2=sstart-1
                    score_local=0
                    for p in range(len(qseq)):
                        aa1=qseq[p]
                        aa2=sseq[p]
                        r1+=aa1!='-'
                        r2+=aa2!='-'
                        if aa1==aa2 and r1 in qacc_site_dict:
                            score_local+=qacc_site_dict[r1]
                    blast_dict[qacc][i]=(sacc,score,score_local,ID,
                        score_local/sum([qacc_site_dict[r1] for r1 in qacc_site_dict]))
                else:
                    blast_dict[qacc][i]=(sacc,score,score,ID,ID)
        return blast_dict
    
    def parse_dl(self,dlfile):
        dl_dict=dict()
        fp=open(dlfile)
        for line in fp.read().splitlines():
            target,ECnumber,cscore=line.split('\t')[:3]
            if not target in dl_dict:
                dl_dict[target]=dict()
            dl_dict[target][ECnumber]=float(cscore)
        fp.close()
        return dl_dict

    def combine_result(self,tsvfile2,tsvfile1,tsvfile):
        target_list=[]
        predict_dict=dict()
        fp=open(tsvfile2)
        for line in fp.read().splitlines():
            items   =line.split('\t')
            target  =items[0]
            ECnumber=items[1]
            if not target in predict_dict:
                predict_dict[target]=''
                target_list.append(target)
            if ECnumber=='0.-.-.-':
                predict_dict[target]+=line+'\n'
        fp.close()
        fp=open(tsvfile1)
        for line in fp.read().splitlines():
            items   =line.split('\t')
            target  =items[0]
            ECnumber=items[1]
            if not target in predict_dict:
                predict_dict[target]=''
                target_list.append(target)
            if ECnumber!='0.-.-.-':
                predict_dict[target]+=line+'\n'
        fp.close()

        txt=''.join([predict_dict[target] for target in target_list])
        fp=open(tsvfile,'w')
        fp.write(txt)
        fp.close()

        dl_dict = self.parse_dl(tsvfile)
        return dl_dict
    
    def get_score(self,target,blast_dict,exp_dict):
        predict_global_dict=dict()
        predict_local_dict=dict()
        denominator_global=0
        denominator_local=0
        weight_global=1
        if target in blast_dict:
            for sacc,score_global,score_local,IDglobal,IDlocal in blast_dict[target]:
                if not sacc in exp_dict:
                    continue
                denominator_global+=score_global
                denominator_local +=score_local
                weight_global*=(1-IDglobal)
                for ECnumber in exp_dict[sacc]:
                    if not ECnumber in predict_global_dict:
                        predict_global_dict[ECnumber]=0
                        predict_local_dict[ECnumber]=0
                    predict_global_dict[ECnumber]+=score_global
                    predict_local_dict[ECnumber] +=score_local
        weight_dynamic=1-weight_global
        return predict_global_dict,predict_local_dict,denominator_global,denominator_local,weight_dynamic
    
    def write_blast(self,blast_dict,exp_dict,tsvfile):
        target_list=sorted(set([target for target in blast_dict]))
        print("writing %s for %d target"%(tsvfile,len(target_list)))
        txt_seq2=''
        for t,target in enumerate(target_list):
            if t%1000==0:
                print("predict %d %s"%(t+1,target))
            predict_blast_global_dict,predict_blast_local_dict,denominator_blast_global,denominator_blast_local,weight_blast=self.get_score(target,blast_dict,exp_dict)
            ECnumber_list=[ECnumber for ECnumber in predict_blast_global_dict]
            ECnumber_list=list(set(ECnumber_list))
            predict_seq2_list=[]
            for ECnumber in ECnumber_list:
                score_blast_global=0
                score_blast_local =0
                if ECnumber in predict_blast_global_dict:
                    score_blast_global=predict_blast_global_dict[ECnumber]
                    score_blast_local =predict_blast_local_dict[ECnumber]
                    if denominator_blast_global>0:
                        score_blast_global/=denominator_blast_global
                    if denominator_blast_local>0:
                        score_blast_local/=denominator_blast_local
                cscore_blast=weight_blast*score_blast_global+(1.-weight_blast)*score_blast_local
                predict_seq2_list.append((cscore_blast,ECnumber))
            predict_seq2_list.sort(reverse=True)
            for cscore,ECnumber in predict_seq2_list:
                if cscore<0.001:
                    break
                txt_seq2+="%s\t%s\t%.3f\n"%(target,ECnumber,cscore)
        fp=open(tsvfile,'w')
        fp.write(txt_seq2)
        fp.close()
        return

    def write_output(self,blast1_dict,blast_dict,foldseek1_dict,foldseek_dict,
        exp_dict,dl_dict):
        target_list=[target for target in blast_dict
                  ]+[target for target in foldseek_dict
                  ]+[target for target in dl_dict]
        target_list=sorted(set(target_list))
        print("writing %s for %d target"%(self.result_file,len(target_list)))
        txt=''
        txt_seq1=''
        txt_seq2=''
        txt_str=''
        txt_template=''
        weight_method=0.9
        weight_dl=0.3
        for t,target in enumerate(target_list):
            if t%1000==0:
                print("predict %d %s"%(t+1,target))
            predict_blast_global_dict,predict_blast_local_dict,denominator_blast_global,denominator_blast_local,weight_blast=self.get_score(target,blast1_dict,exp_dict)
            predict_foldseek_global_dict,predict_foldseek_local_dict,denominator_foldseek_global,denominator_foldseek_local,weight_foldseek=self.get_score(target,foldseek1_dict,exp_dict)

            ECnumber_list = [ECnumber for ECnumber in predict_blast_global_dict
                          ]+[ECnumber for ECnumber in predict_foldseek_global_dict]
            if target in dl_dict:
                ECnumber_list+=[ECnumber for ECnumber in dl_dict[target]]
            ECnumber_list=list(set(ECnumber_list))

            weight_method = weight_blast
        
            predict_list=[]
            predict_seq2_list=[]
            predict_str_list=[]
            predict_template_list=[]
            for ECnumber in ECnumber_list:
                cscore_dl = 0
                if target in dl_dict and ECnumber in dl_dict[target]:
                    cscore_dl = dl_dict[target][ECnumber]
                
                #if ECnumber=="0.-.-.-":
                    #predict_list.append((cscore_dl,ECnumber))
                    #continue

                cscore_blast=0
                if target in blast_dict and ECnumber in blast_dict[target]:
                    cscore_blast=blast_dict[target][ECnumber]
                predict_seq2_list.append((cscore_blast,ECnumber))

                cscore_foldseek=0
                if ECnumber.endswith(".-.-.-"):
                    if target in blast_dict and ECnumber in blast_dict[target]:
                        cscore_foldseek=blast_dict[target][ECnumber]
                else:
                    if target in foldseek_dict and ECnumber in foldseek_dict[target]:
                        cscore_foldseek=foldseek_dict[target][ECnumber]
                predict_str_list.append((cscore_foldseek,ECnumber))

                cscore = weight_method* cscore_blast + (1-weight_method)*cscore_foldseek 
                predict_template_list.append((cscore,ECnumber))
                #weight_dl = 0.5*(1 - weight_blast) #F1=0.7833, 0.9168
                weight_dl = 0.3
                #if weight_blast>0.9: # 0.7853, 0.9170
                    #weight_dl=0

                cscore = weight_dl * cscore_dl + (1-weight_dl)*cscore
                predict_list.append((cscore,ECnumber))
            predict_list.sort(reverse=True)
            predict_seq2_list.sort(reverse=True)
            predict_str_list.sort(reverse=True)
            predict_template_list.sort(reverse=True)
                    
            for cscore,ECnumber in predict_list:
                if cscore<0.001:
                    break
                txt+="%s\t%s\t%.3f\n"%(target,ECnumber,cscore)
            for cscore,ECnumber in predict_seq2_list:
                if cscore<0.001:
                    break
                txt_seq2+="%s\t%s\t%.3f\n"%(target,ECnumber,cscore)
            for cscore,ECnumber in predict_str_list:
                if cscore<0.001:
                    break
                txt_str+="%s\t%s\t%.3f\n"%(target,ECnumber,cscore)
            for cscore,ECnumber in predict_template_list:
                if cscore<0.001:
                    break
                txt_template+="%s\t%s\t%.3f\n"%(target,ECnumber,cscore)
        #fp=open(os.path.join(self.working_dir,"mmseqs.tsv"),'w')
        #fp.write(txt_seq2)
        #fp.close()
        #fp=open(os.path.join(self.working_dir,"foldseek.tsv"),'w')
        #fp.write(txt_str)
        #fp.close()
        fp=open(os.path.join(self.working_dir,"template.tsv"),'w')
        fp.write(txt_template)
        fp.close()
        fp=open(self.result_file,'w')
        fp.write(txt)
        fp.close()
        return

    def main(self):
        tsvfile2 = os.path.join(working_dir,"DL2.tsv")
        tsvfile1 = os.path.join(working_dir,"DL1.tsv")
        tsvfile  = os.path.join(working_dir,"DL.tsv")
        dl_dict  = self.combine_result(tsvfile2, tsvfile1, tsvfile)

        #### run mmseqs ####
        mmseqs1=os.path.join(working_dir,"mmseqs1.m6")
        mmseqs2=os.path.join(working_dir,"mmseqs2.m6")
        tmpdir=os.path.join(working_dir,"tmp")
        if not os.path.isfile(mmseqs1):
            System(settings["mmseqs"]+" easy-search --max-seqs 4 --format-output query,qlen,target,tlen,evalue,bits,alnlen,fident,qaln,qstart,taln,tstart --threads 1 -e 1e-2 "+self.fasta_file+' '+settings['db']+' '+mmseqs1+' '+tmpdir)
        if not os.path.isfile(mmseqs2):
            System(settings["mmseqs"]+" easy-search --max-seqs 4 --format-output query,qlen,target,tlen,evalue,bits,alnlen,fident,qaln,qstart,taln,tstart --threads 1 -e 1e-2 "+self.fasta_file+' '+settings['db2']+' '+mmseqs2+' '+tmpdir)
        infile_list=[mmseqs1,mmseqs2]
        #infile_list=[mmseqs1]
        
        foldseek1=os.path.join(working_dir,"foldseek1.m8")
        foldseek2=os.path.join(working_dir,"foldseek2.m8")
        foldseek1_dict=dict()
        foldseek2_dict=dict()
        if self.pdb_dir:
            cmd=''
            for filename in os.listdir(self.pdb_dir):
                if ".pdb" in filename or ".ent" in filename or ".cif" in filename:
                    cmd+=' '+filename
            if not os.path.isfile(foldseek1):
                System("cd "+self.pdb_dir+";"+settings["foldseek"]+" easy-search "+cmd+" "+settings["afdb"]+" "+foldseek1+" "+tmpdir+" --format-output \"query,qlen,target,tlen,evalue,bits,alnlen,nident,qaln,qstart,taln,tstart,qtmscore,ttmscore\" --tmalign-fast 1 -e 0.1 -s 7.5")
            infile_list.append(foldseek1)
            if not os.path.isfile(foldseek2):
                System("cd "+self.pdb_dir+";"+settings["foldseek"]+" easy-search "+cmd+" "+settings["afdb2"]+" "+foldseek2+" "+tmpdir+" --format-output \"query,qlen,target,tlen,evalue,bits,alnlen,nident,qaln,qstart,taln,tstart,qtmscore,ttmscore\" --tmalign-fast 1 -e 0.1 -s 7.5")
            infile_list.append(foldseek2)
        
        exp_dict,site_dict=self.read_annotation(infile_list)

        blast1_dict=self.parse_blast(mmseqs1,site_dict,infmt='blast')
        blast2_dict=self.parse_blast(mmseqs2,site_dict,infmt='blast')
        tsvfile1   =os.path.join(working_dir, "mmseqs1.tsv")
        tsvfile2   =os.path.join(working_dir, "mmseqs2.tsv")
        self.write_blast(blast1_dict,exp_dict,tsvfile1)
        self.write_blast(blast2_dict,exp_dict,tsvfile2)
        tsvfile    = os.path.join(working_dir,"mmseqs.tsv")
        blast_dict = self.combine_result(tsvfile2, tsvfile1, tsvfile)

        if self.pdb_dir:
            foldseek1_dict=self.parse_blast(foldseek1,site_dict,infmt='foldseek')
            foldseek2_dict=self.parse_blast(foldseek2,site_dict,infmt='foldseek')
            tsvfile1   =os.path.join(working_dir, "foldseek1.tsv")
            tsvfile2   =os.path.join(working_dir, "foldseek2.tsv")
            self.write_blast(foldseek1_dict,exp_dict,tsvfile1)
            self.write_blast(foldseek2_dict,exp_dict,tsvfile2)
            tsvfile    = os.path.join(working_dir,"foldseek.tsv")
            foldseek_dict= self.combine_result(tsvfile2, tsvfile1, tsvfile)
        #dl_dict=self.parse_dl( os.path.join(working_dir,"DL.tsv") )
        #self.write_output(blast_dict,foldseek_dict,exp_dict,dl_dict)
        self.write_output(blast1_dict,blast_dict,
            foldseek1_dict,foldseek_dict,exp_dict,dl_dict)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # example usage: python InterLabelGO_pred.py -w example -f example/example.fasta --use_gpu
    parser.add_argument('-w', '--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('-p', '--pdb_dir', type=str, help='folder for input pdb file(s)', required=False)
    parser.add_argument('-f', '--fasta_file', type=str, help='fasta file', required=True)
    parser.add_argument('-top', '--top_terms', type=int, help='number of top terms to be keeped in the prediction', default=500)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    args = parser.parse_args()
    working_dir = os.path.abspath(args.working_dir)
    fasta_file  = os.path.abspath(args.fasta_file)
    device      = 'cuda' if args.use_gpu else 'cpu'
    pdb_dir     = os.path.abspath(args.pdb_dir) if args.pdb_dir else ''
    
    for idx in ['1','2']:
        result_file=os.path.join(working_dir,"DL"+idx+".tsv")
        if os.path.isfile(result_file):
            continue
        InterLabelGO_pipeline(
            working_dir=working_dir,
            fasta_file=fasta_file,
            device=device,
            top_terms=args.top_terms,
            aspects=['EC'],
            model_dir=settings['MODEL_CHECKPOINT_DIR'+idx],
            result_file=result_file,
        ).main()

    combine_pipeline(
        working_dir=working_dir,
        fasta_file=fasta_file,
        pdb_dir=pdb_dir,
    ).main()
