from os.path import join, dirname
import os

root_dir = os.path.dirname(os.path.abspath(__file__))


"""
This is the settings file for InterLabelGO+
It contains all the default paths
"""
settings_dict = {
    'root_dir': root_dir,
    'DATA_DIR': join(root_dir, 'Data'),
    'ia_file': join(root_dir, 'Data', 'network_training_data', 'IA.txt'),
    'ia_script': join(root_dir, 'utils', 'IC.py'),
    
    'train_ec_tsv':     join(root_dir, 'Data', 'ec_raw_data', 'exp.ec.tsv'),
    'train_ec_fasta':   join(root_dir, 'Data', 'ec_raw_data', 'exp.ec.fasta'),
    'train_terms_tsv':  join(root_dir, 'Data', 'ec_raw_data', 'train_terms.tsv'),
    'train_seqs_fasta': join(root_dir, 'Data', 'ec_raw_data', 'train_seq.fasta'),
    'cdhit_path': join(root_dir, 'utils', 'cd-hit'),
    'parse_isa': join(root_dir, 'utils', 'parse_isa'),

    'esm3b_path': join(root_dir, 'Data', 'esm_models', 'esmc_600m_2024_12_v0.pth'),
    'embedding_dir': join(root_dir, 'Data', 'embeddings'),
    'tmp_dir': join(root_dir, 'Data', 'tmp'),

    'TRAIN_DATA_CLEAN_DIR': join(root_dir, 'Data', 'network_training_data'),
    'alignment_db': join(root_dir, 'Data', 'alignment_database'),
    'alignment_labels': join(root_dir, 'Data', 'alignment_labels'),
    'MODEL_CHECKPOINT_DIR': join(root_dir, 'models', 'ZLPR_PTF1_GOF1'),
    'LOGS_DIR': join(root_dir, 'logs'),
    'SUBMISSION_DIR': join(root_dir, 'submissions')
}

training_config = {
    'activation':'gelu',
    'layer_list':[1024],
    'embed_dim':1152,
    'dropout':0.3,
    'epochs':200,
    'batch_size':512,
    'pred_batch_size':8124*4,
    'learning_rate':0.001,
    'num_models': 5,
    'patience':10,
    'min_epochs':20,
    'seed':1234567890,
    'repr_layers': [34, 35, 36],
    'log_interval':1,
    'eval_interval':1,
    'monitor': 'both',
}

add_res_dict = {
    'EC' :False,
    'EC1':True,
    'EC2':False,
    'EC3':False,
    'EC4':False,
} # whether to add residual connections or not
