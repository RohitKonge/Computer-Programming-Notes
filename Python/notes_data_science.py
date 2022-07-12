  Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell

NOTE - Defining a Lot of variables takes a lot of memory so as we get better at Python we move more and more towards One Liner Code
NOTE - and , or are used for   Single Bool comaprisions 
       &   , |  are used for a Series Bool Comparisions
           
           
---------------------------- PYTHON FOR DATA ANALYSIS - NUMPY ---------------------------

Its a Linear Algebra Library

---------------------    NUMPY ARRAYs     -------------------------

import numpy as np 

we can call randint as ---> from numpy.random import randint

NOTE - All elements in the array will be FLOAT64

arr = np.array([[1,2,3],[4,6,9]])      # ---> Here different length list are Deprecated

print(arr)

print(np.arange( 0, 10, 3))    # (Inclusive, Exclusive, Steps to Jump) Just like Python Range
                               #  Return an array

np.zeros(3)                     ---> Matrix of 3 Zero's
np.zeros(3,4)                   ---> Matrix of 3 rows and 4 columns

np.ones(4)                      ---> Matrix of 4 One's
np.ones(5,6)                    ---> Matrix of 5 rows and 6 columns

np.linspace(12, 23, 14)         ---> Returns a 1D Array of 14 Evenly space elements between 12 and 23
                                 Here (Inclusive, Inclusive)

np.eye(4)                       ---> Returns a 4x4 , Identity Matrix i.e 1 is in the diagonal elements and 0 elsewhe.remove()

print(type(np.eye(4)[0][0]))    ---> FLOAT64

np.random.rand(5,6)             ---> Returns a [5,6] Matrix of value b/w 0 and 1

np.random.randn(3,4)            ---> Returns a [3,4] Matrix of value of a Normal Distribution Curve

np.random.randint(2,100,10)     ---> (Inclusive, Exclusive, Size of Array) , Returns Random int(value, base)

arr.reshape(5,6)                ---> Makes the previous arr into a 5,6 matrix but note that it should have 5*6 = 30 Elements

arr.max()
arr.min()                       ---> Returns Max and Min value of the Array

arr.argmax()
arr.argmin()                    ---> Return the Index of the Max and Min value of the Array

arr.shape

arr = np.random.randint(3,8,9)
print(arr.reshape(3,3).shape)

print(arr.dtype)

---------------------    NUMPY Indexing and Selection     -------------------------

arr = np.arange(3, 10)

We can slice the Array Just Like Slicing List in Python

arr[0:3] = 100                  ---> This sets the value of arr[0] = arr[1] = arr[2] = 100 





























---------------------   Python for Data Analysis - PANDAS     -----------------------

- Pandas is Open Source and built on top of NUMPY

Things to Learn :
  
1. Series & how they interact with Pandas
2. DataFrames of Pandas
3. Work of Missing Data
4. GroupBy of Pandas
5. Merging, Joining and Concatenating DataFrames of Pandas
6. Operations of Pandas
7. Data In and Out of Pandas, Such as CSV, Excel, SQL Files


---------------> 1. Series

import numpy as np
import pandas as pd 

list1 = ['a', 'b', 'c']
list2 = [10, 20, 30]
arr   = np.array(list2)
d     = {'a' : 10, 'b' : 20, 'c' : 30}

print()   # ---> Here list2 is the Label

# we can also use builtin functions as data 

print(pd.Series(data= [sum, len, print], index = list2)) 


a = pd.Series(data= list1, index = list2)

print(a[10])        # Wil Print 'a'

# lets say S1 and S2 are 2 Series Then (S1 + S2) will give Add the Label/Indexs and return NAN if the data is not found in both of them

S1 = pd.Series(data= list1, index = list2)
S2 = pd.Series(data= ['a', 'd', 'c'], index = [20,30,40])

print(S1+S2)

---------------> 2. DataFrames - 1


---> Creating DataFrame

import numpy as np, pandas as pd
from numpy.random import randn

np.random.seed(101)     ---> We set a seed to get the same random nums across different users 

pd.DataFrame(Elements, Row Indices, Column Indices)

df = pd.DataFrame(randn(3,3), ['a','b','c'], [1,2,3])
print(df)

          1         2         3
a  2.706850  0.628133  0.907969
b  0.503826  0.651118 -0.319318
c -0.848077  0.605965 -2.018168



---> Getting Data From DataFrame (Here we can get and also give the data as well)

3 Ways to retrive COLUMN Data

1. print(df[1])       ---> This is recommended --> Will print out a Series of Column 1

2. print(df.1)        ---> Will also print out a Series of Column 1

3. print(df[[1,2]])   ---> calling multiple Columns


NOTE - Calling a Single Column will give a Series 
       Calling Multiple Columns will give a DataFrame    
       
       
2 Ways to retrive ROW Data

1. df.loc[["a","c"], [2,3]]       ---> You can also get individual cell info
2. df.iloc[[1,2],[2,3]]           ---> Here we are using the Indicse of the Rows and Columns



---> Making Extra Columns in DataFrame

df[4] = df[1] + df[2] | randn(3,1)    ---> This will add 1 more column 


---> Deleting Rows and Columns

NOTE - df.shape == (3,3) which is a tuple so (axis = 0 == Rows) and (axis = 1 == Columns)

df.drop([1], axis = 1, inplace = True)  --> here inplace confirms that we want to delete the column permanently if inplace == False then it wont delete it permanently

df.drop(["b"], axis = 0, inplace = True)



---------------------> 3. DataFrames - 2


--->   Conditional Selection

bool_df = df > 0      i.e this returns a DF

print(df > 0)         ---> This will print the DF but with True/False at every cell 

import numpy as np, pandas as pd
from numpy.random import randn

df = pd.DataFrame(randn(4,4))

print(df[ df > 0])    ---> This is same as print(df > 0) but for "false" cells it will write NAN

          0         1         2         3
0       NaN  0.948022  0.641107  0.697116
1  0.769657       NaN  0.086328  0.282669
2  0.191764       NaN       NaN  0.087429
3  0.509276       NaN       NaN       NaN

df[df[0]>0]           ---> This will return the DF but with column 0 elements > 0

NOTE - when we use df[sdfasdf]  here "df" tells that it will return the whole DataFrame


---> To add an index of [0,1,2,3.......n] besides the rows

df.reset_index(inplace=True)    ----> This is also a temporary change so we use inplace

   index         0         1         2         3
0      0 -0.774455 -1.598083 -1.401992 -0.650946
1      1  1.019981  1.059463  0.022018  0.087523
2      2 -1.478124  0.219193 -0.678391  1.314480
3      3  0.208541  0.611723  0.377006  0.921247


df.set_index(3,drop =  False, append = False, inplace = True)
print(df)









