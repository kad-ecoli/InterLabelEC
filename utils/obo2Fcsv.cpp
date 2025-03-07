const char* docstring="\n"
"obo2Fcsv go.obo is_a.csv name.csv alt_id.csv [part_of]\n"
"    convert obo format ontology definition to tab-delimited text\n"
"\n"
"Input:\n"
"    go.obo     - obo format ontology definition\n"
"\n"
"Output:\n"
"    is_a.csv   - GO_ID Aspect is_a_direct is_a_indirect\n"
"    name.csv   - GO_ID Aspect name\n"
"    alt_id.csv - alt_id GO_ID\n"
"\n"
"Option:\n"
"    part_of    - whether to include 'part_of' to find parent term\n"
"                 0: only include 'is_a'\n"
"                 1: (default) include both 'is_a' and 'part_of'\n"
;

#include <iostream>
#include "StringTools.hpp"

using namespace std;

size_t obo2csv(const string &inputfilename, const string &is_a_filename, 
    const string &name_filename, const string &alt_id_filename,
    const int part_of=0)
{
    string line="";
    string GOterm="";
    int    GOterm_int;
    string Aspect="";
    string alt_id="";
    string is_a="";
    string txt="";
    string name="";
    string parent="";
    int    parent_int;
    string GO_pref="";
    int i,j,k;

    vector<string>      GOterm_list;
    map<string,string>  Aspect_dict;
    map<string,vector<string> >alt_id_dict;
    map<string,vector<string> >is_a_dict;
    map<string,string>  name_dict;
    vector<string>      line_vec;
    vector<string>      obsolete_list;
    vector<string>      tmp_list;
    vector<int>         tmp_int_list;
    map<int,vector<int> >is_a_indirect_dict;

    /* read input obo */
    bool fromStdin=(inputfilename=="-");
    bool termStart=false;
    ifstream fin;
    ofstream fout;
    if (!fromStdin) fin.open(inputfilename.c_str());
    while((fromStdin)?cin.good():fin.good())
    {
        if (fromStdin) getline(cin,line);
        else           getline(fin,line);
        if (line.size()==0)
        {
            GOterm="";
            termStart=false;
            continue;
        }

        if (line=="[Term]")
        {
            termStart=true;
            continue;
        }
        if (!termStart) continue;

        if (Startswith(line,"id: "))
        {
            GOterm=line.substr(4);
            GOterm_list.push_back(GOterm);
        }
        else if (Startswith(line,"name: "))
        {
            name_dict[GOterm]=line.substr(6);
        }
        else if (Startswith(line,"namespace: "))
        {
            Aspect=line.substr(11);
            if      (Aspect=="molecular_function") Aspect="F";
            else if (Aspect=="biological_process") Aspect="P";
            else if (Aspect=="cellular_component") Aspect="C";
            Aspect_dict[GOterm]=Aspect;
        }
        else if (Startswith(line,"alt_id: "))
        {
            alt_id=line.substr(8);
            if (alt_id_dict.count(GOterm)==0) alt_id_dict[GOterm]=tmp_list;
            alt_id_dict[GOterm].push_back(alt_id);
        }
        else if (Startswith(line,"is_a: "))
        {
            Split(line,line_vec);
            if (line_vec.size()<=1) continue;
            is_a=line_vec[1];
            if (is_a_dict.count(GOterm)==0) is_a_dict[GOterm]=tmp_list;
            is_a_dict[GOterm].push_back(is_a);
            for (i=0;i<line_vec.size();i++) line_vec[i].clear();
            line_vec.clear();
        }
        else if (line=="is_obsolete: true")
            obsolete_list.push_back(GOterm);
    }
    if (fromStdin) fin.close();

    /* write name */
    txt="";
    for (i=0;i<GOterm_list.size();i++)
    {
        GOterm=GOterm_list[i];
        if (find(obsolete_list.begin(), obsolete_list.end(), GOterm) != 
                 obsolete_list.end()) continue;
        if (Aspect_dict.count(GOterm)==0)
        {
            cerr<<"FATAL ERROR! "<<GOterm<<" lack namespace"<<endl;
            exit(1);
        }
        else if (Aspect_dict[GOterm]!="F") continue;
        name="";
        if (name_dict.count(GOterm)) name=name_dict[GOterm];
            txt+=GOterm+'\t'+Aspect_dict[GOterm]+'\t'+name+'\n';
    }
    if (name_filename=="-") cout<<txt<<flush;
    else
    {
        fout.open(name_filename.c_str());
        fout<<txt<<flush;
        fout.close();
    }

    /* write alt_id */
    txt="";
    for (i=0;i<GOterm_list.size();i++)
    {
        GOterm=GOterm_list[i];
        if (Aspect_dict[GOterm]!="F") continue;
        if (find(obsolete_list.begin(), obsolete_list.end(), GOterm) != 
                 obsolete_list.end()) continue;
        txt+=GOterm+'\t'+GOterm+'\n';
        if (alt_id_dict.count(GOterm))
            for (j=0;j<alt_id_dict[GOterm].size();j++)
                txt+=alt_id_dict[GOterm][j]+'\t'+GOterm+'\n';
    }
    if (alt_id_filename=="-") cout<<txt<<flush;
    else
    {
        fout.open(alt_id_filename.c_str());
        fout<<txt<<flush;
        fout.close();
    }

    /* obtain parent */
    GO_pref=GOterm.substr(0,GOterm.find_first_of(':')+1);
    int digits=GOterm.size()-GO_pref.size();
    for (i=0;i<GOterm_list.size();i++)
    {
        GOterm=GOterm_list[i];
        if (Aspect_dict[GOterm]!="F") continue;
        if (find(obsolete_list.begin(), obsolete_list.end(), GOterm) != 
                 obsolete_list.end()) continue;
        is_a_indirect_dict[GO2int(GOterm)]=tmp_int_list;
        if (is_a_dict.count(GOterm))
        {
             for (j=0;j<is_a_dict[GOterm].size();j++)
                 is_a_indirect_dict[GO2int(GOterm)].push_back(
                                    GO2int(is_a_dict[GOterm][j]));
        }
    }
    int appendNum;
    int layerNum=0;
    while (true)
    {
        layerNum++;
        cerr<<"updating parent level "<<layerNum<<endl;
        appendNum=0;
        for (i=0;i<GOterm_list.size();i++)
        {
            GOterm_int=GO2int(GOterm_list[i]);
            if (is_a_indirect_dict.count(GOterm_int)==0) continue;
            tmp_int_list=is_a_indirect_dict[GOterm_int];
            for (j=0;j<tmp_int_list.size();j++)
            {
                parent_int=tmp_int_list[j];
                if (is_a_indirect_dict.count(parent_int))
                {
                    for (k=0;k<is_a_indirect_dict[parent_int].size();k++)
                    {
                        if (find(is_a_indirect_dict[GOterm_int].begin(),
                                 is_a_indirect_dict[GOterm_int].end(), 
                                 is_a_indirect_dict[parent_int][k]) == 
                                 is_a_indirect_dict[GOterm_int].end())
                        {
                            is_a_indirect_dict[GOterm_int].push_back(
                                 is_a_indirect_dict[parent_int][k]);
                            appendNum++;
                        }
                    }
                }
                else cerr<<"No such parent "<<int2GO(parent_int,GO_pref,digits)<<endl;
            }
            tmp_int_list.clear();
        }
        if (appendNum==0) break;
    }

    /* write parent */
    txt="";
    tmp_list.clear();
    size_t parent_count=0;
    for (i=0;i<GOterm_list.size();i++)
    {
        GOterm=GOterm_list[i];
        if (find(obsolete_list.begin(), obsolete_list.end(), GOterm) != 
                 obsolete_list.end()) continue;
        if (Aspect_dict[GOterm]!="F") continue;
        GOterm_int=GO2int(GOterm);
        if (is_a_dict.count(GOterm))
        {
            for (j=0;j<is_a_indirect_dict[GOterm_int].size();j++)
            {
                parent_int=is_a_indirect_dict[GOterm_int][j];
                parent=int2GO(parent_int,GO_pref,digits);
                if (Aspect_dict[parent]!="F") continue;
                if (find(is_a_dict[GOterm].begin(),is_a_dict[GOterm].end(),
                    parent) == is_a_dict[GOterm].end())
                    tmp_list.push_back(parent);
            }
            txt+=GOterm+'\t'+Aspect_dict[GOterm]+'\t';
            parent_count=0;
            for (j=0;j<is_a_dict[GOterm].size();j++)
            {
                parent=is_a_dict[GOterm][j];
                if (Aspect_dict[parent]!="F") continue;
                parent_count++;
                if (parent_count<=1) txt+=parent;
                else txt+=","+parent;
            }
            txt+='\t'+Join(",",tmp_list)+'\n';
            tmp_list.clear();
        }
        else txt+=GOterm+'\t'+Aspect_dict[GOterm]+"\t\t\n";
    }
    if (is_a_filename=="-") cout<<txt<<flush;
    else
    {
        fout.open(is_a_filename.c_str());
        fout<<txt<<flush;
        fout.close();
    }

    /* clean up */
    int GOtermNum=GOterm_list.size()-obsolete_list.size();
    string ().swap(line);
    string ().swap(GOterm);
    string ().swap(Aspect);
    string ().swap(alt_id);
    string ().swap(is_a);
    string ().swap(txt);
    string ().swap(name);
    string ().swap(parent);
    string ().swap(GO_pref);
    vector<string>      ().swap(GOterm_list);
    map<string,string>  ().swap(Aspect_dict);
    map<string,vector<string> >().swap(alt_id_dict);
    map<string,vector<string> >().swap(is_a_dict);
    map<string,string>  ().swap(name_dict);
    vector<string>      ().swap(line_vec);
    vector<string>      ().swap(obsolete_list);
    vector<string>      ().swap(tmp_list);
    vector<int>         ().swap(tmp_int_list);
    map<int,vector<int> >().swap(is_a_indirect_dict);
    return GOtermNum;
}

int main(int argc,char **argv)
{
    if (argc!=5 && argc!=6)
    {
        cerr<<docstring;
        return 0;
    }

    string inputfilename  =argv[1];
    string is_a_filename  =argv[2];
    string name_filename  =argv[3];
    string alt_id_filename=argv[4];
    int    part_of        =(argc>5)?atoi(argv[5]):1;

    cerr<<obo2csv(inputfilename, is_a_filename,
        name_filename, alt_id_filename, part_of)<<" terms"<<endl;

    string ().swap(inputfilename);
    string ().swap(is_a_filename);
    string ().swap(name_filename);
    string ().swap(alt_id_filename);
    return 0;
}
