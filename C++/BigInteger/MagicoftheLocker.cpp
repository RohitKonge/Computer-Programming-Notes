/* Rohit Konge 
 The Harder You Work The Luckier You Get */

#include<bits/stdc++.h>

using namespace std;

int solve(int n)
{
    long long int  mod = (1e9+7);
    int n1 = n/2;
    int n2 = abs(n-n1);
    int ans = ((n1 % mod)*(n2 % mod)) % mod ;
    cout << ans;
    return ans;
}

int main()
{

    solve(int(1e9));

    return 0;
}