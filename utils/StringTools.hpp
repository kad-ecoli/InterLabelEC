#ifndef StringTools_h
#define StringTools_h 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
//#include <malloc.h>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <iomanip>
#include <map>

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

/* strip white space at the begining or end of string */
string Trim(const string &inputString)
{
    string result = inputString;
    int idxBegin = inputString.find_first_not_of(" \n\r\t");
    int idxEnd = inputString.find_last_not_of(" \n\r\t");
    if (idxBegin >= 0 && idxEnd >= 0)
        result = inputString.substr(idxBegin, idxEnd + 1 - idxBegin);
    return result;
}

bool Startswith(const string &fullString, const string &partialString)
{
    return (fullString.substr(0,partialString.size())==partialString);
}

bool Endswith(const string &fullString, const string &partialString)
{
    return (fullString.size()>=partialString.size() &&
        fullString.substr(fullString.size()-partialString.size())==partialString);
}

string Join(const string &delimiter, const vector<string>&line_vec)
{
    string line="";
    size_t i;
    for (i=0;i<line_vec.size();i++)
    {
        if (i==0) line+=line_vec[0];
        else      line+=delimiter+line_vec[i];
    }
    return line;
}

int GO2int(string &GOterm)
{
    int idxBegin = GOterm.find_first_not_of('0',
                   GOterm.find_first_of(':')+1);
    return atoi(GOterm.substr(idxBegin).c_str());
}

string int2GO(int GOterm_int,const string &GO_pref,int digits)
{
    string GOterm=to_string(GOterm_int);
    while (GOterm.size()<digits)
        GOterm='0'+GOterm;
    GOterm=GO_pref+GOterm;
    return GOterm;
}

#endif
