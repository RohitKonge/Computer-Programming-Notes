/* Rohit Konge 
 The Harder You Work The Luckier You Get */

#include<bits/stdc++.h>

using namespace std;

int main()
{
    int b = 1;
    int a = (b > 0)? 9 : 7 ;  Ternary Operator

    swap(a,b)
    reverse(str.begin, str.end)             --> Reverses the string
    to_string(a)                            --> Converts Int to String


/////////////////////        ARRAY        //////////////////////////

    array<int, 6> arr = {1,2,3,4,5,6};

    arr.at(3);               Get i'th' element

    arr[2];                  Get i'th' element

    arr.size();             

    arr.front();            
    arr.back(); 

     All operations are O(1) 

    for(auto x : arr) { cout << x << endl; }

////////////////////         VECTOR      ///////////////////////////

     Can resize itself
     Uses Dynaminc Memory Allocation
     For every addition it doubles itself (when it reaches multiple of 2)

     We dont have 'push front' because we dont have anything in the front

    vector<int> vec = {9,8,7,6,5};

    vec[2];
    vec.clear();
    vec.push_back(40);
    vec.pop_back();
    vec.reserve(1000);       Reserves 1000 units of space
    vec.size();

    

    vector<int> vec2(4,20);   Fill Contructor ,  4 ints with value 20

    array<int,4> arr = {1,2,3,4};
    vector<int> vec3(arr.begin(), arr.end());    Range Constructor

    int arr2[3] = {1,2,3};
    int n = sizeof(arr2)/sizeof(int); ---->   (4*3)/4    int = 4 bytes
    vector<int> vec5(arr2, arr2 + n );
 
    vector<int> vec4(vec3);             Copy Constructor
    
    vec.size();
    vec.capacity();                 Space reserved for the data
    vec.erase(vec.begin()+1);       erase the 2nd element
    vec.erase(vec.begin() , vec.begin() + 2);   erase the range of first 3 elements

    class notes
    {
    private:
        /* data */
    public:
        notes(/* args */);
        ~notes();
    };
    

    
    Capacity doubles with the size and it becomes a linear
    opeartion and expensive so we use vec.reserve();


////////////////////         DEQUE      ///////////////////////////


    Double ended Queue can Expand and Contract on both ends

    not guaranteed to store elements in continuous locations


    
        Methods
        []
        clear()
        front()
        back()
        pop_back()
        pop_front()
        push_back()
        push_front()
    

    Initializing/Constructing it is the same as vector

    deque<int> deq = {1,2,3,4,5};

    Insertion and Deletion take Linear time


////////////////////       STACK      ///////////////////////////

    stack<string>books;
    books.push("asd");
    books.push("asd");
    books.push("asd");
    books.push("asd");

    while(!books.empty()){
        cout<< books.top();
        books.pop();
    }


////////////////////        QUEUE         ///////////////////////////



    queue<string>books2;
    books2.push("asd");
    books2.push("asd");
    books2.push("asd");
    books2.push("asd");

    while(!books2.empty()){
        cout<< books2.front();
        books2.pop();
    }


////////////////////    PRIORITY QUEUE         ///////////////////////////


    Using it as a Heap Data Structure

    This is in the       #include<queue>       header file

    priority_queue<int> pq;     Greater to Smaller Value

    pq.push(13);
    pq.push(130);
    pq.push(11);    

    class Compare{
        public:
            bool operator()(int a , int b){
                return a < b; 
            }
    };

    priority_queue<int, vector<int>, greater<int>> pq2; Smaller to Greater

    Can also write own Custom Comparerator


////////////////////    BITMANIPULATION       ///////////////////////////

    Bitwise Operators

        AND                &
        OR                 |
        XOR                ^       Exclusive OR
        NOT                ~       ONES COMPLIMENT
        LEFT SHIFT         <<
        RIGHT SHIFT        >>
        
    
    
       XOR 
        0^0  = 0 
        0^1  = 1 
        1^0  = 1 
        1^1  = 0 

        TRICK : If A^B =C then A^C =B and B^C= A  
    

    
      NOT    ~  -->  Flips all the bits
    
    NOTE -  Doing a negation of bit 0 is 1 but negation of int 0 is -1

        Because  ~0  --> 00000000000  -->  1111111111  -> 
        Left most bit tells about the negative sign and all the following 1's
        tells about magnitude so we take 2's compliment and add 1 on the 0th bit
        so it becomes 10000000001 == -1
    

     LEFT AND RIGHT SHIFT
       5 << n ;   5*(2^n)

       5 >> n ;   5/(2^n)

    

      EVEN and ODD
        int x = 5;

        //NOTE - Even numbers have rightmost bit as 0 and 
                 Odd  numbers have rightmost bit as 1
        x is odd if x&1 is true and even if x&1 is false
   
 
      GET, SET, CLEAR ith Bit

    int mask = (1<<i);

    GET   - Use AND &
    SET   - Use OR  |
    CLEAR - Use NOT ~mask and AND &

    UPDATE i'th' bit to v(1 or 0) - first clear that bit and then use 
    OR with (v<<i)
    
    CLEAR last i Bits - int mask(~0 << i)
    n = n & mask

    CLEAR Bits in Range - (int n, int i, int j){
        int a = ~0 << j+1;
        int b = 1<<i -1;
        int mask = a|b;
        n = n & mask;
    }

    

    TRICK to check if N is a power of 2

     if ( N & (N-1) == 0 ) {
        cout << "Power of 2";
    }  

     Counting number of SET Bits
    while(n>0)
    int count += n&1    Tells about the rightmost bit
    n = n>>1
    

    TRICK - 

    while(n>0)
    n = n & (n-1); this removes the last bit which is SET, can also be any number
    count++;

    

    
    
    ////////////////////    BIG INTEGER       ///////////////////////////

    NOTE -  USE PYTHON OR JAVA FOR BIG INTEGER PROBLEMS
    (PYTHON IS BETTER)

    int           = 32 bit , 2^32-1, 10^9       - 4 bytes
    long long int = 64 bit , 2^64-1, 10^18      - 8 bytes
    big integer   = More than 18 digits 

--> Large Addition

    56
+   1098
+   10922

1. We keep the smaller numbers above the larger number
2. Then reverse each number

    65
+   8901
+   22901

3. Then use 'Sum' (== 6+8+2), 'Carry' = Sum / 10,  ans[i] = Sum % 10

4. Do it for the all the digits and then "REVERSE" the answer

ascii value of '0' is 48 and '5' is 53

Now to convert Char to Int use -->  int z = '5' - '0' ( == 53 - 48 = 5)


--> Array And Integer Multiplication

See the code file 'VeryLargeFactorials.cpp'




////////////////////    LINEAR RECURRENCES AND MATRIX EXPONENTIATION      ///////////////////////////


--> Binary Exponentiation

int pow(int a, int b)
{
    int res = 1;
    while(b)
    {
        if(b&1) 
        {
            res *= a;
        }
        a *= a;
        b>>1;
    }

    return res;
}

WE get the binary of the exponent and if the bit is set then multiply the base to it 
Also for a = 10^18 which is a 64 bit number this can be done in 64 steps 


--> Modular Exponentiation 

Just like Binary Exponentiation but we just modulo it with a prime number

int pow(int a, int b)
{
    int res = 1;
    while(b)
    {
        if(b&1) 
        {
            res = ((res % modulo) *(a % modulo)) % modulo;
        }
        a *= a;
        b>>1;
    }
    return res;
}
  return 0;
}