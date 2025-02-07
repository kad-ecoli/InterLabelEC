const char* docstring="\n"
"parse_isa input.tsv output.tsv\n"
"    enfore true path rule\n"
"\n"
"    for EC number q,\n"
"       ub(q) = 1, if parent(q) is empty, i.e. digit(q)=1\n"
"       ub(q) = min { Cparents(q) }, otherwise.\n"
"       lb(q) = 0, if children(q) is empty, i.e. digit(q)=4\n"
"       lb(q) = max { Cchildren(q) }, otherwise.\n"
"    if lb(q)<= C(q)<= ub(q), C(q) no change\n"
"    if lb(q) > C(q) > ub(q), C(q) no change\n"
"    if lb(q) > C(q) < ub(q), C(q) = min{ lb(q), ub(q) }\n"
"    if lb(q) < C(q) > ub(q), C(q) = max{ lb(q), ub(q) }\n"
"    if lb(q) > C(q) = ub(q), C(q) = lb(q)\n"
"    if lb(q) = C(q) > ub(q), C(q) = ub(q)\n"
"\n"
"Input:\n"
"    input.tsv  - function prediction in the following format\n"
"                 [1] target\n"
"                 [2] EC number\n"
"                 [3] C-score\n"
"\n"
"Output:\n"
"    output.tsv - function prediction in the following format\n"
"                 [1] target\n"
"                 [2] EC number\n"
"                 [3] C-score\n"
;

#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <map>
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;

/* split a long string into vectors by whitespace 
 * line          - input string
 * line_vec      - output vector 
 * delimiter     - delimiter */
void Split(const string &line, vector<string> &line_vec,
    const char delimiter=' ',const bool long_delimiter=true)
{
    bool within_word = false;
    for (size_t pos=0;pos<line.size();pos++)
    {
        if (line[pos]==delimiter)
        {
            if (!long_delimiter && within_word == false)
                line_vec.push_back("");
            within_word = false;
            continue;
        }
        if (!within_word)
        {
            within_word = true;
            line_vec.push_back("");
        }
        line_vec.back()+=line[pos];
    }
}

void parse_isa(const string &infile, const string &outfile)
{
    vector<string> input_list;
    map<string,vector<pair<string,double> > >cscore_dict;
    vector<pair<string,double> >tmp_vec;
    map<string,vector<string> >parent_dict;
    
    /* parse input */
    ifstream fin;
    vector<string> line_vec;
    vector<string> ec_vec;
    string line,target,ECnumber,ECparent,ECprefix;
    vector<string> ECnumber_list;
    double cscore;
    int i,j;
    bool fromStdin=(infile=="-");
    if (!fromStdin) fin.open(infile.c_str());
    while((fromStdin)?cin.good():fin.good())
    {
        if (fromStdin) getline(cin,line);
        else           getline(fin,line);
        if (line.size()==0) continue;
        Split(line, line_vec, '\t');
        target  =     line_vec[0];
        ECnumber=     line_vec[1];
        cscore  =atof(line_vec[2].c_str());
        for (i=0;i<line_vec.size();i++) line_vec[i].clear(); line_vec.clear();

        /* parse cscore */
        if (cscore<0 || cscore>1)
        {
            cout<<"WARNING! cscore out of range [0,1]. "
                <<line<<endl;
            if (cscore<0) cscore=0;
            else if (cscore>1) cscore=1;
        }
        if (cscore_dict.count(target)==0)
        {
            input_list.push_back(target);
            cscore_dict[target]=tmp_vec;
        }
        cscore_dict[target].push_back(make_pair(ECnumber,cscore));

        /* parse parent-child */
        if (parent_dict.count(ECnumber)) continue;
        ECnumber_list.push_back(ECnumber);
        Split(ECnumber, ec_vec, '.');
        for (i=3;i>=1;i--)
        {
            if (ec_vec[i]=="-") continue;
            ECparent=ec_vec[0];
            for (j=1;j<i;j++) ECparent+='.'+ec_vec[j];
            for (j=i;j<4;j++) ECparent+=".-";
            line_vec.push_back(ECparent);
            cout<<ECnumber<<" is_a "<<ECparent<<endl;
            break;
        }
        for (i=0;i<ec_vec.size();i++) ec_vec[i].clear(); ec_vec.clear();
        parent_dict[ECnumber]=line_vec;
        for (i=0;i<line_vec.size();i++) line_vec[i].clear(); line_vec.clear();
    }
    if (!fromStdin) fin.close();
    cout<<ECnumber_list.size()<<" EC number from "
        <<input_list.size()<<" target in "<<infile<<endl;

    /* enforce true path rule */
    stringstream buf;
    size_t a,p,round;
    map<string,double> tmp_dict;
    map<string,double> lb_dict;
    map<string,double> ub_dict;
    vector<pair<double,string> >ECnumber_cscore_list;
    int adjustNum=0;
    double lb,ub;
    //double diffcscore=0;
    for (a=0;a<input_list.size();a++)
    {
        target=input_list[a];
        for (i=0;i<cscore_dict[target].size();i++)
        {
            ECnumber=cscore_dict[target][i].first;
            tmp_dict[ECnumber]=cscore_dict[target][i].second;
            //cout<<target<<'\t'<<cscore_dict[target][i].first<<endl;
        }
        //cout<<target<<endl;
        for (round=1;round<=4;round++)
        {
            for (i=0;i<cscore_dict[target].size();i++)
            {
                ECnumber=cscore_dict[target][i].first;
                lb_dict[ECnumber]=0;
                ub_dict[ECnumber]=1;
            }
            for (i=0;i<cscore_dict[target].size();i++)
            {
                ECnumber=cscore_dict[target][i].first;
                cscore=tmp_dict[ECnumber];
                if (parent_dict.count(ECnumber))
                {
                    for (p=0;p<parent_dict[ECnumber].size();p++)
                    {
                        ECparent=parent_dict[ECnumber][p];
                        if (tmp_dict.count(ECparent)==0) continue;
                        if (lb_dict[ECparent]<cscore)
                            lb_dict[ECparent]=cscore;
                        if (ub_dict[ECnumber]>tmp_dict[ECparent])
                            ub_dict[ECnumber]=tmp_dict[ECparent];
                    }
                }
            }
            adjustNum=0;
            for (i=0;i<cscore_dict[target].size();i++)
            {
                ECnumber=cscore_dict[target][i].first;
                cscore  =cscore_dict[target][i].second;
                lb      =lb_dict[ECnumber];
                ub      =ub_dict[ECnumber];
                if (cscore<lb || cscore>ub)
                {
                    adjustNum++;
                    if (lb>cscore && cscore<ub) cscore=MIN(lb,ub);
                    else if (lb<cscore && cscore>ub) cscore=MAX(lb,ub);
                    else if (lb>cscore && cscore==ub) cscore=lb;
                    else if (lb==cscore && cscore>ub) cscore=ub;
                    cscore_dict[target][i].second=cscore;
                    
                }
            }
            if (adjustNum==0) break;
            cout<<"fixing "<<adjustNum<<" terms for "<<target<<" round "<<round<<endl;
            for (i=0;i<cscore_dict[target].size();i++)
            {
                ECnumber=cscore_dict[target][i].first;
                tmp_dict[ECnumber]=cscore_dict[target][i].second;
            }
        }
        for (i=0;i<cscore_dict[target].size();i++)
        {
            ECnumber=cscore_dict[target][i].first;
            cscore  =cscore_dict[target][i].second;
            //if (parent_dict.count(ECnumber));
            //cscore+= 1E-8*(1-count(ECnumber.begin(), ECnumber.end(), '-')/3.);
            ECnumber_cscore_list.push_back(make_pair(cscore,ECnumber));
            //cout<<target<<'\t'<<ECnumber<<'\t'<<cscore<<endl;
        }
        sort(ECnumber_cscore_list.begin(),ECnumber_cscore_list.end());
        for (i=ECnumber_cscore_list.size()-1;i>=0;i--)
        {
            cscore  =ECnumber_cscore_list[i].first;
            ECnumber=ECnumber_cscore_list[i].second;
            if (cscore<0.001) continue;
            //cout<<target<<'\t'<<ECnumber<<'\t'<<cscore<<endl;
            buf<<target<<'\t'<<ECnumber<<'\t'
               <<setiosflags(ios::fixed)<<setprecision(3)<<cscore<<endl;
            ECnumber_cscore_list[i].second.clear();
        }
        ECnumber_cscore_list.clear();
        tmp_dict.clear();
        lb_dict.clear();
        ub_dict.clear();
    }

    /* write output */
    bool toStdout=(outfile=="-");
    if (toStdout) cout<<buf.str()<<flush;
    else
    {
        ofstream fout;
        fout.open(outfile.c_str());
        fout<<buf.str()<<flush;
        fout.close();
    }
    
    /* clean up */
    buf.str(string());
    vector<string> ().swap(input_list);
    map<string,vector<pair<string,double> > >().swap(cscore_dict);
    vector<pair<string,double> >().swap(tmp_vec);
    map<string,vector<string> >().swap(parent_dict);
    vector<string> ().swap(line_vec);
    map<string,double> ().swap(tmp_dict);
    map<string,double> ().swap(tmp_dict);
    map<string,double> ().swap(lb_dict);
    map<string,double> ().swap(ub_dict);
    vector<pair<double,string> >().swap(ECnumber_cscore_list);
    return;
}

int main(int argc,char **argv)
{
    if (argc!=3)
    {
        cerr<<docstring<<endl;;
        return 0;
    }

    string infile    = argv[1];
    string outfile   = argv[2];
    
    parse_isa(infile, outfile);

    string ().swap(infile);
    string ().swap(outfile);
    return 0;
}
