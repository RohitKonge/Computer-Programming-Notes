/* Rohit Konge 
 The Harder You Work The Luckier You Get */

#include<bits/stdc++.h>

using namespace std;

int solve(vector<int> nums, int k)
{
  // vector<int> rin = {a,b,c};
  // sort(rin.begin(), rin.end()-1);
  // int rin[3] = {a,b,c};
  // sort(rin, rin+3);

  // int ans;

  // if((rin[0] - rin[2]) == 0){
  //   ans = rin[0];
  // }else{
  //   ans = rin[0];
  //   int sub = rin[2]-rin[0];
  //   rin[2] = sub;
  //   ans += (rin[1] < rin[2]) ? rin[1] : rin[2] ;
  // }

  // return ans;

  // int sub1 = abs(rin[2]-rin[0]);
  // rin[0] = 0;
  // rin[2] = sub1;

  // int sub2 = abs(rin[1]- sub1);

  // if(sub1 == 0){
  //   ans =  rin[0];
  // }else if (sub2 == 0){
  //   ans =  (rin[0] + rin[1]);
  // }else {
  //   ans =  (rin[0] + ((sub1 > rin[1]) ? rin[1]: sub1));
  // }

  // if((rin[0] != rin[1]) and (rin[1] != rin[2]) ){
  //   ans = rin[0] + rin[1];
  // }else if (rin[0] == rin[1]){
    
  // }else if(rin[1] == rin[2]){
  //   ans = rin[1];
  // }else if(rin[0] == rin[2]){
  //   ans = rin[0];
  // }

  //cout<< ans;

  // nums = {3,2,3,1,2,4,5,5,6};
  // k = 4;
  
  set<int,greater<int>> uniq_nums;
  
  int ans = 0;

  for (int i : nums)
  {
    uniq_nums.insert(i);
  }
  int check = 1;
  for(auto j: uniq_nums){
    if(check==k){
      ans = j;
      break;
    }else{
      check++;
    }
  }
  cout<< ans;
} 

int main()
{

    // #ifndef ONLINE_JUDGE
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    // #endif


    solve({-3,-2,-3,-1,-2,-4,-5,-5,-6},4);

    return 0;
}