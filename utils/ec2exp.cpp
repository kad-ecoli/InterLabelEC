const char* docstring=""
"ec2exp uniprot.ec.tsv exp.ec.tsv iea.ec.tsv\n"
"\n"
"Input:\n"
"    uniprot.ec.tsv - accession, ECnumber, evidence, ChEBI, BINDING, ACT_SITE\n"
"\n"
"Output:\n"
"    exp.ec.tsv     - annotation supported by experimental evidence\n"
"    iea.ec.tsv     - annotation not supported by experimental evidence\n"
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


int getECprefix(const string&ECnumber, string&ECprefix)
{
    int digit=0;
    if      (Endswith(ECnumber,".-.-.-")) 
    {
        ECprefix=ECnumber.substr(0,ECnumber.size()-5);
        digit=1;
    }
    else if (Endswith(ECnumber,".-.-")) 
    {
        ECprefix=ECnumber.substr(0,ECnumber.size()-3);
        digit=2;
    }
    else if (Endswith(ECnumber,".-"))
    {
        ECprefix=ECnumber.substr(0,ECnumber.size()-1);
        digit=3;
    }
    else
    {
        ECprefix=ECnumber;
        digit=4;
    }
    return digit;
}

void uniq_ec(const vector<string> &ECnumber_list, map<string, string> &exp_dict,
    map<string, string> &iea_dict, map<string, string> &site_dict, 
    string &accession, string &exp_line, string &iea_line)
{
    exp_line.clear();
    iea_line.clear();
    size_t i,j;
    int haschild=0;
    string ECnumber,ECprefix,ECchild;
    if (ECnumber_list.size()>1)
    {
        /* remove parent EC if child EC exist and has better evidence */
        for (i=0;i<ECnumber_list.size();i++)
        {
            ECnumber=ECnumber_list[i];
            if (exp_dict.count(ECnumber)==0 && iea_dict.count(ECnumber)==0)
                continue;
            if (ECnumber.back()!='-') continue;
            getECprefix(ECnumber,ECprefix);
            for (j=0;j<ECnumber_list.size();j++)
            {
                if (i==j) continue;
                ECchild=ECnumber_list[j];
                if (Startswith(ECchild,ECprefix) && 
                    (exp_dict.count(ECchild) || iea_dict.count(ECnumber)))
                {
                    if (exp_dict.count(ECnumber)) exp_dict.erase(ECnumber);
                    if (iea_dict.count(ECnumber)) iea_dict.erase(ECnumber);
                    break;
                }
            }
        }

        /* upgrade child EC evidence if parent EC is experimental */
        for (i=0;i<ECnumber_list.size();i++)
        {
            ECnumber=ECnumber_list[i];
            if (exp_dict.count(ECnumber)==0 || ECnumber.back()!='-') continue;
            if (getECprefix(ECnumber,ECprefix)<3) continue;
            haschild=0;
            for (j=0;j<ECnumber_list.size();j++)
            {
                if (i==j) continue;
                ECchild=ECnumber_list[j];
                if (Startswith(ECchild,ECprefix) && iea_dict.count(ECchild) 
                    && iea_dict[ECchild].find("BRENDA")!=string::npos
                    )
                {
                    exp_dict[ECchild]=exp_dict[ECnumber];
                    if (exp_dict[ECchild].size()) exp_dict[ECchild]+=",BRENDA";
                    else exp_dict[ECchild]="BRENDA";
                    iea_dict.erase(ECchild);
                    haschild++;
                }
            }
            if (haschild) exp_dict.erase(ECnumber);
        }

        /* transfer child EC site information to parent */
        for (i=0;i<ECnumber_list.size();i++)
        {
            ECnumber=ECnumber_list[i];
            if (ECnumber.back()!='-' || site_dict[ECnumber].size()>2) continue;
            getECprefix(ECnumber,ECprefix);
            for (j=0;j<ECnumber_list.size();j++)
            {
                if (i==j) continue;
                ECchild=ECnumber_list[j];
                if (Startswith(ECchild,ECprefix) &&
                    site_dict[ECnumber].size()<site_dict[ECchild].size())
                    site_dict[ECnumber]=site_dict[ECchild];
            }
        }
    }
    for (i=0;i<ECnumber_list.size();i++)
    {
        ECnumber=ECnumber_list[i];
        if      (exp_dict.count(ECnumber)) exp_line+=accession+'\t'+
            ECnumber+'\t'+exp_dict[ECnumber]+'\t'+site_dict[ECnumber]+'\n';
        else if (iea_dict.count(ECnumber)) iea_line+=accession+'\t'+
            ECnumber+'\t'+iea_dict[ECnumber]+'\t'+site_dict[ECnumber]+'\n';
    }
    return;
}


size_t ec2exp(const string &tsvfile="", const string &expfile="-", 
    const string &ieafile="-")
{
    /* CAFA5 evidence code */
    vector<string> evidence_code_list;
    evidence_code_list.push_back(",EXP,");
    evidence_code_list.push_back(",IDA,");
    evidence_code_list.push_back(",IPI,");
    evidence_code_list.push_back(",IMP,");
    evidence_code_list.push_back(",IGI,");
    evidence_code_list.push_back(",IEP,");

    evidence_code_list.push_back(",HTP,");
    evidence_code_list.push_back(",HDA,");
    evidence_code_list.push_back(",HMP,");
    evidence_code_list.push_back(",HGI,");
    evidence_code_list.push_back(",HEP,");

    evidence_code_list.push_back(",TAS,");
    evidence_code_list.push_back(",IC,");

    evidence_code_list.push_back(",ECO:0000269,"); // EXP
    evidence_code_list.push_back(",ECO:0000305,"); // IC

    /* read dat */
    ofstream fp_exp;
    ofstream fp_iea;
    if (expfile!="-")
    {
        fp_exp.open(expfile.c_str(),ofstream::out);
        fp_exp<<"#accession\tECnumber\tevidence\tChEBI\tBINDING\tACT_SITE"<<endl;
    }
    else  cout<<"#accession\tECnumber\tevidence\tChEBI\tBINDING\tACT_SITE"<<endl;
    if (ieafile!="-")
    {
        fp_iea.open(ieafile.c_str(),ofstream::out);
        fp_iea<<"#accession\tECnumber\tevidence\tChEBI\tBINDING\tACT_SITE"<<endl;
    }
    else  cout<<"#accession\tECnumber\tevidence\tChEBI\tBINDING\tACT_SITE"<<endl;
    
   
    vector<string> split_vec;
    map<string,string> exp_dict; // ECnumber => evidence
    map<string,string> iea_dict; // ECnumber => evidence
    map<string,string> site_dict; // ECnumber => ChEBI,BINDING,ACT_SITE
    vector<string> ECnumber_list;
    string accession="";
    string ECnumber="";
    string evidence="";
    string exp_line,iea_line;
    string line;
    ifstream fp_in;
    bool isexp;
    size_t i;

    if (tsvfile!="-") fp_in.open(tsvfile.c_str(),ofstream::out);
    while ((tsvfile!="-")?fp_in.good():cin.good())
    {
        if (tsvfile!="-") getline(fp_in,line);
        else getline(cin,line);
        if (line.size()==0 || line[0]=='#') continue;

        Split(line,split_vec,'\t',false);
        if (split_vec[0]!=accession)
        {
            if (ECnumber_list.size())
            {
                uniq_ec(ECnumber_list,exp_dict,iea_dict,
                    site_dict,accession,exp_line,iea_line);
                if (expfile=="-") cout<<exp_line<<flush;
                else            fp_exp<<exp_line<<flush;
                if (ieafile=="-") cout<<iea_line<<flush;
                else            fp_iea<<iea_line<<flush;
            }
            map<string,string>().swap(exp_dict);
            map<string,string>().swap(iea_dict);
            map<string,string>().swap(site_dict);
            vector<string>().swap(ECnumber_list);
        }
        accession=split_vec[0];
        ECnumber =split_vec[1];
        evidence =','+split_vec[2]+',';
        isexp=0;
        if (split_vec[2].size())
        {
            for (i=0;i<evidence_code_list.size();i++)
            {
                if (evidence.find(evidence_code_list[i])==string::npos)
                    continue;
                isexp=true;
                break;
            }
        }
        if (isexp) exp_dict[ECnumber]=split_vec[2];
        else iea_dict[ECnumber]=split_vec[2];
        site_dict[ECnumber]=split_vec[3]+'\t'+split_vec[4]+'\t'+split_vec[5];
        ECnumber_list.push_back(ECnumber);

        for (i=0;i<split_vec.size();i++) split_vec[i].clear(); split_vec.clear();
        //vector<string>().swap(split_vec);
    }
    if (accession.size())
    {
        if (ECnumber_list.size())
        {
            uniq_ec(ECnumber_list,exp_dict,iea_dict,
                site_dict,accession,exp_line,iea_line);
            if (expfile=="-") cout<<exp_line<<flush;
            else            fp_exp<<exp_line<<flush;
            if (ieafile=="-") cout<<iea_line<<flush;
            else            fp_iea<<iea_line<<flush;
        }
        map<string,string>().swap(exp_dict);
        map<string,string>().swap(iea_dict);
        map<string,string>().swap(site_dict);
        vector<string>().swap(ECnumber_list);
    }
    
    if (tsvfile!="-")   fp_in.close();
    if (expfile!="-")   fp_exp.close();
    if (ieafile!="-")   fp_iea.close();
    return 0;
}

int main(int argc, char **argv)
{
    /* parse commad line argument */
    if(argc!=4)
    {
        cerr<<docstring;
        return 0;
    }
    string tsvfile    =argv[1];
    string expfile    =argv[2];
    string ieafile    =argv[3];
    ec2exp(tsvfile, expfile, ieafile);
    return 0;
}
