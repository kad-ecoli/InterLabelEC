const char* docstring=""
"zcat uniprot.dat.gz | uniprot2noec is_a.csv - exp.nonec.tsv iea.nonec.tsv exp.nonec.fasta\n"
"\n"
"Input:\n"
"    is_a.csv            - GO term, Aspect, direct, indirect\n"
"    uniprot.dat.gz      - text format uniprot annotation\n"
"\n"
"Output:\n"
"    exp.nonec.tsv   - accession, GO term, GO term name, experimental GO term evidence\n"
"    iea.nonec.tsv   - accession, GO term, GO term name, GO term evidence\n"
"    exp.nonec.fasta - FASTA sequence of non-enzyme protein\n"
;

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include "StringTools.hpp"

using namespace std;

size_t uniprot2nonec(const string &isafile="", const string &infile="-",
    const string &tsvfile="", const string &ieafile="", const string &fastafile="-")
{
    map<string, bool> GOterm_dict;
    map<string, bool> CatalyticTerm_dict;
    CatalyticTerm_dict["GO:0003824"]=1; // catalytic activity
    CatalyticTerm_dict["GO:0022857"]=1; // transmembrane transporter activity
    map<string, bool> IonBinding_dict;
    IonBinding_dict["GO:0046872"]=1; // metal ion binding
    string line;
    string GOterm;
    ifstream fp_in;
    if (isafile!="-") fp_in.open(isafile.c_str(),ios::in);
    while ((isafile!="-")?fp_in.good():cin.good())
    {
        if (isafile!="-") getline(fp_in,line);
        else getline(cin,line);
        if (line.size()>10)
        {
            GOterm_dict[line.substr(0,10)]=1;
            if (line.find("GO:0003824")!=string::npos ||
                line.find("GO:0022857")!=string::npos)
                CatalyticTerm_dict[line.substr(0,10)]=1;
            else if (line.find("GO:0046872")!=string::npos)
                IonBinding_dict[line.substr(0,10)]=1;
        }
    }
    if (isafile!="-") fp_in.close();
    cerr<<CatalyticTerm_dict.size()<<" out of "<<GOterm_dict.size()
        <<" GO terms are catalytic activity"<<endl;
    cerr<<IonBinding_dict.size()<<" out of "<<GOterm_dict.size()
        <<" GO terms are ion binding"<<endl;

    /* CAFA5 evidence code */
    vector<string> evidence_code_list;
    evidence_code_list.push_back("EXP");
    evidence_code_list.push_back("IDA");
    evidence_code_list.push_back("IPI");
    evidence_code_list.push_back("IMP");
    evidence_code_list.push_back("IGI");
    evidence_code_list.push_back("IEP");

    evidence_code_list.push_back("HTP");
    evidence_code_list.push_back("HDA");
    evidence_code_list.push_back("HMP");
    evidence_code_list.push_back("HGI");
    evidence_code_list.push_back("HEP");

    evidence_code_list.push_back("TAS");
    evidence_code_list.push_back("IC");

    /* read dat */
    ofstream fp_tsv;
    ofstream fp_iea;
    ofstream fp_fasta;
    size_t nseqs=0;
    size_t i;
    vector<string> lines;
    int has_EC=0;
    int has_GOterm=0;
    int has_iea_GOterm=0;
    int SQ=0;
    string sequence;
    string accession;
    string tsv_txt;
    string iea_txt;
    vector<string> split_vec;
    if (tsvfile!="-")
    {
        fp_tsv.open(tsvfile.c_str(),ofstream::out);
        fp_tsv<<"#accession\tGOterm\tname\tevidence"<<endl;
    }
    else  cout<<"#accession\tGOterm\tname\tevidence"<<endl;

    if (ieafile.size())
    {
        if (ieafile!="-")
        {
            fp_iea.open(ieafile.c_str(),ofstream::out);
            fp_iea<<"#accession\tGOterm\tname\tevidence"<<endl;
        }
        else  cout<<"#accession\tGOterm\tname\tevidence"<<endl;
    }

    if (fastafile!="-") fp_fasta.open(fastafile.c_str(),ofstream::out);

    if (infile!="-") fp_in.open(infile.c_str(),ios::in);
    while ((infile!="-")?fp_in.good():cin.good())
    {
        if (infile!="-") getline(fp_in,line);
        else getline(cin,line);

        if (Startswith(line,"//"))
        {
            if (!has_EC && (has_GOterm || has_iea_GOterm))
            {
                if (ieafile.size())
                {
                    if (ieafile=="-") cout<<iea_txt<<flush;
                    else fp_iea<<iea_txt<<flush;
                }
                if (has_GOterm)
                {
                    if (tsvfile=="-") cout<<tsv_txt<<flush;
                    else fp_tsv<<tsv_txt<<flush;
                    if (fastafile=="-") cout<<'>'<<accession<<'\n'<<sequence<<'\n';
                    else fp_fasta<<'>'<<accession<<'\n'<<sequence<<'\n';
                }
            }
            vector<string>().swap(lines);
            has_EC=0;
            has_GOterm=0;
            has_iea_GOterm=0;
            SQ=0;
            sequence.clear();
            accession.clear();
            tsv_txt.clear();
            iea_txt.clear();
        }
        else
        {
            if (Startswith(line,"AC   ") && accession.size()==0)
            {
                Split(line.substr(5),split_vec,';');
                accession=split_vec[0];
                vector<string>().swap(split_vec);
            }
            else if (Startswith(line,"DE            EC="))
                has_EC++;
            else if (Startswith(line,"DR   BRENDA;"))
                has_EC++;
            else if (Startswith(line,"CC   -!- CATALYTIC ACTIVITY:"))
                has_EC++;
            else if (Startswith(line,"DR   GO; GO:"))
            {
                GOterm=line.substr(9,10);
                if (GOterm_dict.count(GOterm)==0 || 
                    GOterm=="GO:0005524" || // ATP binding
                    GOterm=="GO:0042802" || // identical protein binding
                    GOterm=="GO:0005515" || // protein binding
                    GOterm=="GO:0051260" || // protein homooligomerization
                    GOterm=="GO:0003676" || // nucleic acid bining (but RNA/DNA binding counts)
                    IonBinding_dict.count(GOterm))
                    continue;
                if (CatalyticTerm_dict.count(GOterm))
                    has_EC++;
                else
                {
                    for (i=0;i<evidence_code_list.size();i++)
                    {
                        if (line.find("; "+evidence_code_list[i]+':')==
                            string::npos) continue;
                        has_GOterm++;
                        Split(line.substr(23),split_vec,';');
                        tsv_txt+=accession+'\t'+GOterm+'\t'+split_vec[0]+'\t'+evidence_code_list[i]+'\n';
                        vector<string>().swap(split_vec);
                        break;
                    }
                    
                    if (ieafile.size() && line.find("; IBA:")==string::npos)
                    {
                        has_iea_GOterm++;
                        Split(line.substr(23),split_vec,';');
                        iea_txt+=accession+'\t'+GOterm+'\t'+split_vec[0]+'\t';
                        line=split_vec[1].substr(1);
                        vector<string>().swap(split_vec);
                        Split(line,split_vec,':');
                        iea_txt+=split_vec[0]+'\n';
                        vector<string>().swap(split_vec);
                    }
                }
            }
            else if (Startswith(line,"SQ   "))
            {
                SQ=1;
            }
            else if (Startswith(line,"     ") && SQ)
            {
                sequence+=line.substr(5,10);
                if (line.size()<16) continue;
                sequence+=line.substr(16,10);
                if (line.size()<27) continue;
                sequence+=line.substr(27,10);
                if (line.size()<38) continue;
                sequence+=line.substr(38,10);
                if (line.size()<49) continue;
                sequence+=line.substr(49,10);
                if (line.size()<60) continue;
                sequence+=line.substr(60,10);
            }
        }
    }
    
    if (tsvfile!="-")   fp_tsv.close();
    if (ieafile.size() && ieafile!="-") fp_iea.close();
    if (fastafile!="-") fp_fasta.close();


    map<string, bool> ().swap(GOterm_dict);
    map<string, bool> ().swap(CatalyticTerm_dict);
    map<string, bool> ().swap(IonBinding_dict);
    string ().swap(line);
    string ().swap(GOterm);
    vector<string> ().swap(evidence_code_list);
    vector<string> ().swap(lines);
    string ().swap(sequence);
    string ().swap(accession);
    vector<string> ().swap(split_vec);
    return nseqs;
}

int main(int argc, char **argv)
{
    /* parse commad line argument */
    if(argc!=5 && argc!=6)
    {
        cerr<<docstring;
        return 0;
    }
    string isafile    =argv[1];
    string infile     =argv[2];
    string tsvfile    =argv[3];
    string ieafile    ="";
    string fastafile  =argv[4];
    if (argc>5)
    {
        ieafile       =argv[4];
        fastafile     =argv[5];
    }
    uniprot2nonec(isafile, infile, tsvfile, ieafile, fastafile);
    return 0;
}
