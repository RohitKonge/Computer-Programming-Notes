/* Rohit Konge 
 The Harder You Work The Luckier You Get */

#include<bits/stdc++.h>

using namespace std;



void solve(string a, string b)
{
    int n = a.length(), p = b.length();
    if(n > p)
    {
        swap(a, b);
    }else
    {
        swap(n, p);
    }

    reverse(a.begin(), a.end());
    reverse(b.begin(), b.end());

    int carry = 0, sum = 0;
    string ans;

    for(int i = 0; i < n; i++)
    {
        if(i < p)
        {
            sum = (a[i] - '0' )+ (b[i] - '0') + carry;
        }else
        {
            sum = (b[i] - '0') + carry;
        }

        carry = sum/10;
        ans.append(to_string(sum%10)) ;       
        sum = 0;
    }
    if (carry > 0)
    {
        /* code */
        ans.append(to_string(carry));
    }
    reverse(ans.begin(), ans.end());

    cout << ans;
}

int main()
{

    // #ifndef ONLINE_JUDGE
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    // #endif

    solve("3452100000", "876");

    return 0;
}