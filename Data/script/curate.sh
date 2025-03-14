#!/bin/bash
FILE=`readlink -e $0`
curdir=`dirname $FILE`
dbdir=`dirname $curdir`
rawdir="$dbdir/raw"
curatedir="$dbdir/ec_raw_data"
bindir=`readlink -e $dbdir/../utils/`

mkdir -p $rawdir
cd $rawdir
$bindir/obo2Fcsv go.obo is_a.csv name.csv alt_id.csv
$bindir/go2ec.py is_a.csv ec2go go2ec.tsv
time zcat uniprot_sprot.dat.gz | $bindir/uniprot2ec go2ec.tsv rhea2ec.tsv - uniprot_sprot.ec.tsv  uniprot_sprot.ec.fasta
time $bindir/ec2exp uniprot_sprot.ec.tsv  exp_sprot.ec.tsv  iea_sprot.ec.tsv

time zcat uniprot_trembl.dat.gz| $bindir/uniprot2ec go2ec.tsv rhea2ec.tsv - uniprot_trembl.ec.tsv uniprot_trembl.ec.fasta
time $bindir/ec2exp uniprot_trembl.ec.tsv exp_trembl.ec.tsv iea_trembl.ec.tsv


time zcat uniprot_sprot.dat.gz | $bindir/uniprot2nonec is_a.csv - exp_sprot.nonec.tsv  iea_sprot.nonec.tsv  exp_sprot.nonec.fasta
time zcat uniprot_trembl.dat.gz| $bindir/uniprot2nonec is_a.csv - exp_trembl.nonec.tsv iea_trembl.nonec.tsv exp_trembl.nonec.fasta


mkdir -p $curatedir
cd $curatedir
cat $rawdir/exp_sprot.nonec.tsv    $rawdir/exp_trembl.nonec.tsv   > $curatedir/exp.nonec.tsv
cat $rawdir/exp_sprot.nonec.fasta  $rawdir/exp_trembl.nonec.fasta > $curatedir/exp.nonec.fasta
cat $rawdir/exp_sprot.ec.tsv       $rawdir/exp_trembl.ec.tsv      > $curatedir/exp.ec.tsv
cat $rawdir/uniprot_sprot.ec.fasta $rawdir/uniprot_trembl.ec.fasta| $bindir/fasta2miss $curatedir/exp.ec.tsv - - $curatedir/exp.ec.fasta



cd $curatedir
echo make alphafold database

version=v4

grep -v '^#' $curatedir/exp.ec.tsv|cut -f1|uniq|sort|uniq > $rawdir/exp.ec.list
cat $rawdir/afdb_uniprot_$version.list $rawdir/exp.ec.list | sort | uniq -c | grep ' 2 ' | grep -ohP "\S+$" > $rawdir/afdb_subset.list
cat $rawdir/afdb_subset.list |sed "s/^/AF-/g"|sed "s/$/-F1-model_$version/g" > $rawdir/id_list.txt
$bindir/mmseqs/bin/mmseqs createsubdb --subdb-mode 0 --id-mode 1 $rawdir/id_list.txt $rawdir/afdb_uniprot_$version $rawdir/afdb_subset
$bindir/foldcomp decompress $rawdir/afdb_subset $rawdir/afdb_subset_pdb
rm $rawdir/exp.ec.list $rawdir/id_list.txt
for target in `cat $rawdir/afdb_subset.list`;do
   mv $rawdir/afdb_subset_pdb/AF-$target-F1-model_$version.pdb $rawdir/afdb_subset_pdb/$target.pdb
done
$bindir/foldseek createdb $rawdir/afdb_subset_pdb/ $curatedir/afdb
$bindir/foldseek createindex $curatedir/afdb $rawdir/afdb_tmp/
rm -rf $rawdir/afdb_tmp
ln -sf afdb_h        $curatedir/afdb_ss_h       
ln -sf afdb_h.dbtype $curatedir/afdb_ss_h.dbtype
ln -sf afdb_h.index  $curatedir/afdb_ss_h.index 
