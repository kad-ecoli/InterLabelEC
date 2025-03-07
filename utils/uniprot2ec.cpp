const char* docstring=""
"zcat uniprot.dat.gz|uniprot2ec go2ec.tsv rhea2ec.tsv - uniprot.ec.tsv uniprot.ec.fasta\n"
"\n"
"Input:\n"
"    go2ec.tsv      - GO term, list of EC number\n"
"    rhea2ec.tsv    - rhea ID, UN, rhea ID, EC number\n"
"    uniprot.dat.gz - text format uniprot annotation\n"
"\n"
"Output:\n"
"    uniprot.ec.tsv   - accession, EC number, evidence, site\n"
"    uniprot.ec.fasta - FASTA sequence of protein with EC number\n"
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
            
void parse_CatalyticActivity(const string &CatalyticActivity_line,
    vector<string> &ec_list, map<string,string> &ec_evidence_dict,
    map<string,string> &ec_ChEBI_dict, map<string,string> &ec_BINDING_dict,
    map<string,string> &ec_ACT_dict, map<string,vector<string> > &rhea2ec_dict)
{
    vector<string> split_vec;
    string ChEBI_line=",";
    string ECnumber;
    string Rhea;
    string ChEBI;
    vector<string> ECO_vec;
    string ECO;
    size_t i;
    Split(CatalyticActivity_line,split_vec,' ');
    for (i=0;i<split_vec.size();i++)
    {
        if (Startswith(split_vec[i],"Xref=Rhea:RHEA:"))
        {
            Rhea=split_vec[i].substr(15);
            if (Rhea.back()==';' || Rhea.back()==',')
                Rhea=Rhea.substr(0,Rhea.size()-1);
        }
        else if (Startswith(split_vec[i],"ChEBI:CHEBI:"))
        {
            ChEBI=split_vec[i].substr(12);
            if (ChEBI.back()==';' || ChEBI.back()==',')
                ChEBI=ChEBI.substr(0,ChEBI.size()-1);
            ChEBI_line+=ChEBI+',';
        }
        else if (Startswith(split_vec[i],"EC="))
        {
            ECnumber=split_vec[i].substr(3);
            if (ECnumber.back()==';' || ECnumber.back()==',')
                ECnumber=ECnumber.substr(0,ECnumber.size()-1);
        }
        else if (Startswith(split_vec[i],"Evidence={ECO:"))
            ECO_vec.push_back(split_vec[i].substr(10,11));
        else if (Startswith(split_vec[i],"ECO:"))
            ECO_vec.push_back(split_vec[i].substr(0,11));
    }
    if (ECnumber.size()==0 && ec_list.size() && Rhea.size() && rhea2ec_dict.count(Rhea))
    {
        for (i=0;i<rhea2ec_dict[Rhea].size();i++)
            if (ec_evidence_dict.count(rhea2ec_dict[Rhea][i]))
                ECnumber=rhea2ec_dict[Rhea][i];
    }
    ECO=Join(",",ECO_vec);
    if (ECnumber.size()==0)
    {
        //cerr<<"WARNING! No EC: "<<CatalyticActivity_line<<endl;
        ;
    }
    else
    {
        if (ec_evidence_dict.count(ECnumber)==0)
        {
            ec_evidence_dict[ECnumber]=ECO;
            ec_ChEBI_dict[ECnumber]=ChEBI_line;
            ec_BINDING_dict[ECnumber]="";
            ec_ACT_dict[ECnumber]="";
            ec_list.push_back(ECnumber);
        }
        else
        {
            /* evidence */
            if (ECO_vec.size())
            {
                if (ec_evidence_dict[ECnumber].size()==0)
                    ec_evidence_dict[ECnumber]=ECO;
                else
                {
                    for (i=0;i<ECO_vec.size();i++)
                        if (ec_evidence_dict[ECnumber].find(ECO_vec[i])==string::npos)
                            ec_evidence_dict[ECnumber]+=','+ECO_vec[i];
                }
            }
            /* ChEBI */
            if (ChEBI_line.size())
            {
                if (ec_ChEBI_dict[ECnumber].size()==0)
                    ec_ChEBI_dict[ECnumber]=ChEBI_line;
                else ec_ChEBI_dict[ECnumber]+=ChEBI_line.substr(1);
            }
        }
    }

    /*clean up */
    vector<string>().swap(split_vec);
    string ().swap(ChEBI_line);
    string ().swap(ECnumber);
    string ().swap(Rhea);
    string ().swap(ChEBI);
    vector<string> ().swap(ECO_vec);
    string ().swap(ECO);
}

void parse_lines(const vector<string> &lines, string &tsv_line,
    string &fasta_line, map<string,string> &go2ec_dict,
    map<string,vector<string> >&rhea2ec_dict)
{
    string accession;
    vector<string> split_vec;
    map<string,string> ec_evidence_dict;
    map<string,string> ec_ChEBI_dict;
    map<string,string> ec_BINDING_dict;
    map<string,string> ec_ACT_dict;
    size_t i,j;
    string line;
    string ECnumber;
    string GOterm;
    string ECO;
    string BINDING;
    string ChEBI;
    vector<string> ec_list;
    string CatalyticActivity_line;
    for (i=0;i<lines.size();i++)
    {
        line=lines[i];
        if (CatalyticActivity_line.size() && !Startswith(line,"CC   "))
        {
            parse_CatalyticActivity(CatalyticActivity_line,ec_list,
                ec_evidence_dict,ec_ChEBI_dict,ec_BINDING_dict,ec_ACT_dict,rhea2ec_dict);
            CatalyticActivity_line.clear();
        }

        if (Startswith(line,"AC   ") && accession.size()==0)
        {
            Split(line.substr(5),split_vec,';');
            accession=split_vec[0];
            vector<string> ().swap(split_vec);
        }
        else if (Startswith(line,"DE            EC="))
        {
            Split(line.substr(14),split_vec,' ');
            if (Startswith(split_vec[0],"EC="))
            {
                if (split_vec[0].back()==';')
                    ECnumber=split_vec[0].substr(3,split_vec[0].size()-4);
                else
                    ECnumber=split_vec[0].substr(3,split_vec[0].size()-3);
                ec_evidence_dict[ECnumber]="";
                ec_ChEBI_dict[ECnumber]   ="";
                ec_BINDING_dict[ECnumber] ="";
                ec_ACT_dict[ECnumber]     ="";
                ec_list.push_back(ECnumber);
                if (split_vec[0].back()!=';')
                {
                    for (j=1;j<split_vec.size();j++)
                    {
                        if (split_vec[j][0]=='{')
                            ECO=split_vec[j].substr(1,11);
                        else ECO=split_vec[j].substr(0,11);
                        if (ec_evidence_dict[ECnumber].size()==0)
                            ec_evidence_dict[ECnumber]=ECO;
                        else if (ec_evidence_dict[ECnumber].find(ECO)==string::npos)
                            ec_evidence_dict[ECnumber]+=','+ECO;
                    }
                }
            }
            vector<string> ().swap(split_vec);
        }
        else if (Startswith(line,"DR   BRENDA;"))
        {
            Split(line.substr(13),split_vec,';');
            ECnumber=split_vec[0];
            if (ECnumber.find('B')==string::npos)
            {
                if (ec_evidence_dict.count(ECnumber)==0)
                {
                    ec_evidence_dict[ECnumber]="BRENDA";
                    ec_ChEBI_dict[ECnumber]   ="";
                    ec_BINDING_dict[ECnumber] ="";
                    ec_ACT_dict[ECnumber]     ="";
                    ec_list.push_back(ECnumber);
                }
                else
                    ec_evidence_dict[ECnumber]+=",BRENDA";
            }
            vector<string> ().swap(split_vec);
        }
        else if (Startswith(line,"CC   "))
        {
            if (Startswith(line,"CC   -!- CATALYTIC ACTIVITY:"))
            {
                if (CatalyticActivity_line.size())
                {
                    parse_CatalyticActivity(CatalyticActivity_line,ec_list,
                        ec_evidence_dict,ec_ChEBI_dict,ec_BINDING_dict,
                        ec_ACT_dict,rhea2ec_dict);
                    CatalyticActivity_line.clear();
                }
            }
            else CatalyticActivity_line+=line.substr(8);
        }
        else if (Startswith(line,"DR   GO; GO:"))
        {
            GOterm=line.substr(9,10);
            if (go2ec_dict.count(GOterm))
            {
                Split(line,split_vec,' ');
                ECO=split_vec.back();
                vector<string> ().swap(split_vec);
                Split(ECO,split_vec,':');
                ECO=split_vec[0];
                vector<string> ().swap(split_vec);
                for (j=0;j<ec_list.size();j++)
                {
                    ECnumber=ec_list[j];
                    if (go2ec_dict[GOterm].find(','+ECnumber+','))
                    {
                        if (ec_evidence_dict[ECnumber].size()==0) ec_evidence_dict[ECnumber]=ECO;
                        else ec_evidence_dict[ECnumber]+=','+ECO;
                    }
                }
            }
        }
        else if (Startswith(line,"FT   "))
        {
            if (Startswith(line,"FT   ACT_SITE"))
            {
                Split(line,split_vec,' ');
                for (j=0;j<ec_list.size();j++)
                    ec_ACT_dict[ec_list[j]]+=','+split_vec[2];
                vector<string> ().swap(split_vec);
            }
            else if ((Startswith(line,"FT   BINDING") ||
                      Startswith(line,"FT   REGION "))
                    && line.find(':')==string::npos)
            {
                BINDING.clear();
                for (j=21;j<line.size();j++)
                    if (line[j]=='.' || ('0'<=line[j] && line[j]<='9'))
                        BINDING+=line[j];
            }
            else if (BINDING.size() && Startswith(line,
                "FT                   /ligand_id=\"ChEBI:CHEBI:"))
            {
                ChEBI=line.substr(45);
                ChEBI=ChEBI.substr(0,ChEBI.size()-1);
                //cerr<<"ChEBI="<<ChEBI<<" line="<<line<<endl;
                for (j=0;j<ec_list.size();j++)
                {
                    ECnumber=ec_list[j];
                    //cerr<<"ec_ChEBI_dict["<<ECnumber<<"]="<<ec_ChEBI_dict[ECnumber]<<endl;
                    if (ec_ChEBI_dict[ECnumber].find(','+ChEBI+',')==string::npos)
                        continue;
                    if (ec_BINDING_dict[ECnumber].size())
                        ec_BINDING_dict[ECnumber]+=','+BINDING;
                    else ec_BINDING_dict[ECnumber]=BINDING;
                    BINDING.clear();
                    break;
                }
            }
            else if (BINDING.size() && (
                Startswith(line,"FT                   /ligand=\"substrate")     ||
                Startswith(line,"FT                   /note=\"Substrate")       ||
                Startswith(line,"FT                   /ligand_note=\"catalytic")||
                Startswith(line,"FT                   /ligand_note=\"substrate")))
            {
                for (j=0;j<ec_list.size();j++)
                {
                    ECnumber=ec_list[j];
                    if (ec_BINDING_dict[ECnumber].size())
                        ec_BINDING_dict[ECnumber]+=','+BINDING;
                    else ec_BINDING_dict[ECnumber]=BINDING;
                }
                BINDING.clear();
            }
        }
        else if (Startswith(line,"     "))
        {
            fasta_line+=line.substr(5,10);
            if (line.size()<16) continue;
            fasta_line+=line.substr(16,10);
            if (line.size()<27) continue;
            fasta_line+=line.substr(27,10);
            if (line.size()<38) continue;
            fasta_line+=line.substr(38,10);
            if (line.size()<49) continue;
            fasta_line+=line.substr(49,10);
            if (line.size()<60) continue;
            fasta_line+=line.substr(60,10);
        }
    }
    fasta_line='>'+accession+'\n'+fasta_line+'\n';
    string site_line;
    string ChEBI_line;
    string ECO_line;
    for (i=0;i<ec_list.size();i++)
    {
        ECnumber=ec_list[i];

        if (ec_evidence_dict[ECnumber].size())
        {
            Split(ec_evidence_dict[ECnumber],split_vec,',');
            ECO_line=split_vec[0];
            for (j=1;j<split_vec.size();j++)
                if (ECO_line.find(split_vec[j])==string::npos)
                    ECO_line+=','+split_vec[j];
            vector<string>().swap(split_vec);
        }
        
        if (ec_ChEBI_dict[ECnumber].size()) 
            ChEBI_line=ec_ChEBI_dict[ECnumber].substr(1,ec_ChEBI_dict[ECnumber].size()-2);
        
        if (ec_ACT_dict[ECnumber].size())
        {
            Split(ec_ACT_dict[ECnumber].substr(1),split_vec,',');
            site_line=',';
            for (j=0;j<split_vec.size();j++)
            {
                site_line.find(','+split_vec[j]+',')==string::npos;
                site_line+=split_vec[j]+',';
            }
            vector<string> ().swap(split_vec);
            site_line=site_line.substr(1,site_line.size()-2);
        }
        
        tsv_line+=accession+'\t'+ECnumber+'\t'+ECO_line+'\t'+ChEBI_line+ 
            '\t'+ec_BINDING_dict[ECnumber]+'\t'+site_line+'\n';
        ECO_line.clear();
        ChEBI_line.clear();
        site_line.clear();
    }

    vector<string>().swap(split_vec);
    map<string,string>().swap(ec_evidence_dict);
    map<string,string>().swap(ec_ChEBI_dict);
    map<string,string>().swap(ec_BINDING_dict);
    map<string,string>().swap(ec_ACT_dict);
    vector<string>().swap(ec_list);
    return;
}

size_t uniprot2ec(const string &go2ecfile="", const string &rhea2ecfile="",
    const string &infile="-", const string &tsvfile="-", const string &fastafile="-")
{
    /* go -> ec mapping */
    map<string,string> go2ec_dict;
    string line;
    ifstream fp_in;
    if (go2ecfile!="-") fp_in.open(go2ecfile.c_str(),ios::in);
    while ((go2ecfile!="-")?fp_in.good():cin.good())
    {
        if (go2ecfile!="-") getline(fp_in,line);
        else getline(cin,line);
        if (line.size()>11)
            go2ec_dict[line.substr(0,10)]=','+line.substr(11)+',';
    }
    if (go2ecfile!="-") fp_in.close();

    /* rhea -> ec mapping */
    map<string,vector<string> > rhea2ec_dict;
    vector<string> split_vec;
    vector<string> tmp_vec;
    if (rhea2ecfile!="-") fp_in.open(rhea2ecfile.c_str(),ios::in);
    while ((rhea2ecfile!="-")?fp_in.good():cin.good())
    {
        if (rhea2ecfile!="-") getline(fp_in,line);
        else getline(cin,line);
        if (line.size() && !Startswith(line,"RHEA_ID"))
        {
            Split(line,split_vec,'\t');
            if (split_vec.size()>=4)
            {
                if (rhea2ec_dict.count(split_vec[0])==0)
                    rhea2ec_dict[split_vec[0]]=tmp_vec;
                rhea2ec_dict[split_vec[0]].push_back(split_vec[3]);
            }
            vector<string>().swap(split_vec);
        }
    }
    if (rhea2ecfile!="-") fp_in.close();

    /* read dat */
    ofstream fp_tsv;
    ofstream fp_fasta;

    size_t nseqs=0;
    size_t i;
    vector<string> lines;
    int has_EC=0;
    int RecName=0;
    int CatalyticActivity=0;
    int binding=0;
    int SQ=0;
    string tsv_line,fasta_line;
    tsv_line="#accession\tECnumber\tevidence\tChEBI\tBINDING\tACT_SITE\n";
    if (tsvfile=="-") cout<<tsv_line;
    else 
    {
        fp_tsv.open(tsvfile.c_str(),ofstream::out);
        fp_tsv<<tsv_line;
    }
    tsv_line.clear();
    if (fastafile!="-") fp_fasta.open(fastafile.c_str(),ofstream::out);

    if (infile!="-") fp_in.open(infile.c_str(),ios::in);
    while ((infile!="-")?fp_in.good():cin.good())
    {
        if (infile!="-") getline(fp_in,line);
        else getline(cin,line);

        if (Startswith(line,"//"))
        {
            if (has_EC)
            {
                //cerr<<'['<<nseqs<<']'<<endl;
                parse_lines(lines,tsv_line,fasta_line, go2ec_dict, rhea2ec_dict);
                nseqs++;
                //cerr<<"tsv_line="<<tsv_line<<endl;
                //cerr<<"fasta_line="<<fasta_line<<endl;

                if (tsvfile=="-") cout<<tsv_line;
                else fp_tsv<<tsv_line;
                tsv_line.clear();
                
                if (fastafile=="-") cout<<fasta_line;
                else fp_fasta<<fasta_line;
                fasta_line.clear();
            }
            vector<string>().swap(lines);
            has_EC=0;
            RecName=0;
            CatalyticActivity=0;
            binding=0;
            SQ=0;
        }
        else
        {
            if (Startswith(line,"AC   "))
            {
                lines.push_back(line);
            }
            else if (Startswith(line,"DE   "))
            {
                if (Startswith(line,"DE   RecName:"))
                {
                    lines.push_back(line);
                    RecName=1;
                }
                else if (Startswith(line,"DE           ") && RecName)
                {
                    lines.push_back(line);
                    has_EC+=(line.find(" EC=")!=string::npos);
                }
                else
                    RecName=0;
            }
            else if (Startswith(line,"CC   "))
            {
                if (Startswith(line,"CC   -!- CATALYTIC ACTIVITY:"))
                {
                    lines.push_back(line);
                    CatalyticActivity=1;
                }
                else if (Startswith(line,"CC       ") && CatalyticActivity)
                {
                    lines.push_back(line);
                    has_EC+=(line.find(" EC=")!=string::npos);
                }
                else
                    CatalyticActivity=0;
            }
            else if (Startswith(line,"FT   "))
            {
                if (Startswith(line,"FT   BINDING"))
                {
                    lines.push_back(line);
                    binding=1;
                }
                else if (Startswith(line,"FT   REGION "))
                {
                    lines.push_back(line);
                    binding=1;
                }
                else if (Startswith(line,"FT   ACT_SITE"))
                {
                    lines.push_back(line);
                    binding=2;
                }
                else if (Startswith(line,"FT          ") && binding)
                    lines.push_back(line);
                else
                    binding=0;
            }
            else if (Startswith(line,"DR   BRENDA;"))
            {
                lines.push_back(line);
                has_EC+=1;
            }
            else if (Startswith(line,"DR   GO; GO:") && 
                line.find("; F:")!=string::npos)
                lines.push_back(line);
            else if (Startswith(line,"SQ   "))
            {
                //lines.push_back(line);
                SQ=1;
            }
            else if (Startswith(line,"     ") && SQ)
                lines.push_back(line);
        }
    }
    
    if (tsvfile=="-") cout<<flush;
    else
    {
        fp_tsv<<flush;
        fp_tsv.close();
    }
    
    if (fastafile=="-") cout<<fasta_line;
    else 
    {
        fp_fasta<<fasta_line;
        fp_fasta.close();
    }
    return nseqs;
}

int main(int argc, char **argv)
{
    /* parse commad line argument */
    if(argc!=6)
    {
        cerr<<docstring;
        return 0;
    }
    string go2ecfile  =argv[1];
    string rhea2ecfile=argv[2];
    string infile     =argv[3];
    string tsvfile    =argv[4];
    string fastafile  =argv[5];
    uniprot2ec(go2ecfile,rhea2ecfile,infile,tsvfile,fastafile);
    return 0;
}
