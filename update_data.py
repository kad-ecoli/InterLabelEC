import os
file_dir = os.path.dirname(os.path.realpath(__file__))
from settings import settings_dict as settings


def main():
    """
    Download the newest UNIPROT GOA file and preprocess to the csv and fasta
    """
    tmp_dir = os.path.join(file_dir, settings['tmp_dir'])
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # remove the old obo file if exists
    aspect_map_dict = {
        "1": "EC1",
        "2": "EC2",
        "3": "EC3",
        "4": "EC4",
    }
    # only grep exp data
    if not os.path.isfile(settings['train_ec_fasta']):
        print("ERROR! no such file "+settings['train_ec_fasta'])
        exit(1)
    if not os.path.isfile(settings['train_ec_tsv']):
        print("ERROR! no such file "+settings['train_ec_tsv'])
        exit(1)

    print("prepare "+settings['train_seqs_fasta'])
    totalN=0
    seleN=0
    seq_list=[]
    target_list=[]
    fp=open(settings['train_ec_fasta'])
    for block in ('\n'+fp.read()).split('\n>'):
        if not block.strip():
            continue
        lines=block.splitlines()
        header=lines[0].split()[0]
        sequence=''.join(lines[1:])
        totalN+=1
        if len(sequence)<30 or len(sequence)>1024:
            continue
        seleN+=1
        seq_list.append((len(sequence),'>'+header+'\n'+sequence+'\n'))
        target_list.append(header)
    fp.close()
    seq_list.sort(reverse=True)
    target_set=set(target_list)
    print("writing %d out of %d sequence"%(seleN,totalN))
    txt=''.join([seq for L,seq in seq_list])
    fp=open(settings['train_seqs_fasta'],'w')
    fp.write(txt)
    fp.close()

    #cmd=settings['cdhit_path']+" -i "+settings['train_seqs_fasta']+" -o "+settings['train_seqs_fasta']+".cdhit -c 0.4 -M 5000 -n 2 -T 16 -g 1 -sc 1"
    #cmd=settings['cdhit_path']+" -i "+settings['train_seqs_fasta']+" -o "+settings['train_seqs_fasta']+".cdhit -c 0.5 -M 5000 -n 2 -T 16 -g 1 -sc 1"
    cmd=settings['cdhit_path']+" -i "+settings['train_seqs_fasta']+" -o "+settings['train_seqs_fasta']+".cdhit -c 0.6 -M 5000 -n 3 -T 16 -g 1 -sc 1"
    #cmd=settings['cdhit_path']+" -i "+settings['train_seqs_fasta']+" -o "+settings['train_seqs_fasta']+".cdhit -c 0.7 -M 5000 -n 4 -T 16 -g 1 -sc 1"
    #cmd=settings['cdhit_path']+" -i "+settings['train_seqs_fasta']+" -o "+settings['train_seqs_fasta']+".cdhit -c 0.8 -M 5000 -n 5 -T 16 -g 1 -sc 1"
    print(cmd)
    os.system(cmd)
    
    print("prepare "+settings['train_terms_tsv'])
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
    fp=open(settings['train_terms_tsv'],'w')
    fp.write(txt)
    fp.close()
    return

if __name__ == '__main__':
    main()
