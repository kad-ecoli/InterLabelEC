#!/bin/bash
FILE=`readlink -e $0`
bindir=`dirname $FILE`
dbdir=`dirname $bindir`
rawdir="$dbdir/raw"

mkdir -p $rawdir
cd $rawdir
wget ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2023_02/knowledgebase/knowledgebase2023_02.tar.gz -O knowledgebase2023_02.tar.gz
tar -xvf knowledgebase2023_02.tar.gz
rm       knowledgebase2023_02.tar.gz
wget http://release.geneontology.org/2024-04-24/ontology/external2go/ec2go -O ec2go
wget http://release.geneontology.org/2024-04-24/ontology/go-basic.obo -O go.obo
wget https://ftp.expasy.org/databases/rhea/tsv/rhea2ec.tsv -O rhea2ec.tsv

version=v4
if [ ! -s "$rawdir/afdb_uniprot_$version" ];then
    wget https://foldcomp.steineggerlab.workers.dev/afdb_uniprot_$version.dbtype -O afdb_uniprot_$version.dbtype
    wget https://foldcomp.steineggerlab.workers.dev/afdb_uniprot_$version.index  -O afdb_uniprot_$version.index
    wget https://foldcomp.steineggerlab.workers.dev/afdb_uniprot_$version.lookup -O afdb_uniprot_$version.lookup
    wget https://foldcomp.steineggerlab.workers.dev/afdb_uniprot_$version.source -O afdb_uniprot_$version.source
    wget https://foldcomp.steineggerlab.workers.dev/afdb_uniprot_$version        -O afdb_uniprot_$version
    cut -f2 -d- raw/afdb_uniprot_$version.lookup | sort| uniq  > afdb_uniprot_$version.list
fi
