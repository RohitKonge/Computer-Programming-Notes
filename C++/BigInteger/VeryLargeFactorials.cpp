/* Rohit Konge 
 The Harder You Work The Luckier You Get */

#include<bits/stdc++.h>

using namespace std;

vector<int> multi(vector<int> a, int b)
{
    int multiplied, carry = 0; 
    vector<int> prev_fact;
    prev_fact.reserve(1000);
    reverse(a.begin(), a.end());
    for (int i = 0;  i < a.size(); i++)
    {
        multiplied = (b * a[i]) + carry;;
        carry = (multiplied / 10);
        prev_fact.push_back(multiplied % 10);
    }
    while(carry>0)
    {
        prev_fact.push_back(carry % 10);
        carry = (carry / 10);
    }
    reverse(prev_fact.begin(), prev_fact.end());
    return prev_fact;
}

void solve(int n)
{
    vector<int> ans = {1};
    ans.reserve(1000);

    for (int i = 1; i < n; i++)
    {
      ans =  multi(ans, i+1);
    }
    for(auto x: ans){
        cout << x;
    }
    
}

int main()
{

    solve(100);

    return 0;
}