Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell

--------------------------------------------------------> Index <------------------------------------------------------------------









---------------------------------------------> PYTHON FOR DATA ANALYSIS - NUMPY <--------------------------------------------------



Its a Linear Algebra Library

np.NaN == Not A Number

Aggregate Function == A function that takes in lots of individual values and then returns a single value


For 'axis = None' , We iterate over all the elements of the numpy array/ multidimensional array
                    Indexing goes from 0 to the last element of the numpy array/ multidimensional array
                    
                    
    'axis = 0 / 1', We iterate over all the elements of the row/ column of the numpy array/ multidimensional array
                    Indexing  == arr[3[4]]



------------------------------->     NUMPY ARRAYs     <-----------------------------------------



import numpy as np 

Numpy Arrays differ from Python List because of its ability to Broadcast

Eg. 

a = np.array([1,2,3,4])
b = np.array([5])
print(a*b)                                      -----> Broadcast

we can call randint as ---> from numpy.random import randint

NOTE -  1. All elements in the array will be FLOAT64

        2. When we use np.''''''(arr)           -----> We have to use the name of the array
        
        
        
arr = np.array([[1,2,3],[4,6,9]])      # ---> Here different length list are Deprecated

print(arr)

print(np.arange( 0, 10, 3))    # (Inclusive, Exclusive, Steps to Jump) Just like Python Range
#  Returns an array

np.arange(50)                   ---> Returns a list from 0 to 49



----------------> Making Matrixes



np.zeros(3)                     ---> Matrix of 3 Zero's
np.zeros(3,4)                   ---> Matrix of 3 rows and 4 columns

np.ones(4)                      ---> Matrix of 4 One's
np.ones(5,6)                    ---> Matrix of 5 rows and 6 columns

np.full((3,4), 11)              ---> Matrix of 3 rows and 4 columns and all the elements == 11

np.linspace(12, 23, 14)         ---> Returns a 1D Array of 14 Evenly space elements between 12 and 23
                                 Here (Inclusive, Inclusive)

np.eye(4)                       ---> Returns a 4x4 , Identity Matrix i.e 1 is in the diagonal elements and 0 elsewhe.remove()

print(type(np.eye(4)[0][0]))    ---> FLOAT64



----------------> Generating Random Numbers

np.random.state(101)            ---> Ensure that the same random numbers are generated every time we run the code

np.random.rand(5,6)             ---> Returns a [5,6] Matrix of value b/w 0 and 1

np.random.randn(3,4)            ---> Returns a [3,4] Matrix of value of a Normal Distribution Curve

np.random.randint(2,100,10)     ---> (Inclusive, Exclusive, Size of Array) , Returns Random int(value, base)
np.random.randint(3,10,(4,5))    ---> Returns a [4,5] Matrix and the value of the elements is between (3,10)



----------------> Some Numpy Functions


arr.reshape(5,6)                ---> Makes the previous arr into a 5,6 matrix but note that it should have 5*6 = 30 Elements

arr.max()
arr.min()                       ---> Returns Max and Min value of the Array

arr.argmax()
arr.argmin()                    ---> Return the Index of the Max and Min value of the Array

arr.shape

arr = np.random.randint(3,8,9)
print(arr.reshape(3,3).shape)     ----> This is way to make a 1D Array to N dimensional Array

print(arr.dtype)                ---> Returns the data type of the elements



----------------> Question on  Numpy Arrays



Q. Find the index of a particular element

Ans ---> np.where('array_name' == np.NaN)                        



Q. Delete a particular element

Ans ---> np.delete('array_name', "index_of_element_to_be_deleted", axis = 0 / 1 / None)

Note --->   For axis = None it follow the index of the elements

            For axis = 0 / 1 it will delete the entire row / column & index_of_element_to_be_deleted = index_of_(row/column)



Q. Check if a value is in a Numpy array

Ans ---> 'value' in 'array_name'        # Returns a Bool



Q. Change a value in a array/ Matrix 


Ans ---> arr[1]     = 23                
         arr[2][4]  = 87            # These will simply change the values



import numpy as np

arr = np.array([[11,2,34], [36,12,87]])
arr2 = np.delete(arr, 4)
print(np.where(arr == 36, axis = None))



------------------------------>     The Basics of NumPy Arrays      <-----------------------------



--------------------->  1. Making a Copy



arr2 = arr1                 ---> This creates a reference to arr1, so chaning arr2 will change arr1 as well
arr2 = np.copy(arr1)        ---> This creates a different COPY of arr1, so changing arr2 wont change arr1

NOTE - This Concept is also TRUE for MultiDimensional Array/ Matrixes



--------------------->  2. Indexing



arr[2]                          ---> Returns the 3rd Element

arr[-1]                         ---> Returns the Last Element

arr[3][2]  (this is a matrix)   ---> Returns the 4th Rows 3rd Column Element



--------------------->  3. SLicing                      (Inclusive, Exclusive, Step)



NOTE - We can use Slicing to Set the Values as Well



For 1D arrays:


    arr[1:3]

    arr[1:3:2]

    arr[1:]

    arr[1::2]

    arr[::2]

    arr[::-1]                       ---> Reverses the Array       

    arr[5::-2]



For MultiDimensional Arrays :
    
    arr[1:][::2]                    ----> For All rows, from 1 to end of Matrix , For All columns from start to end with a step of 2

    arr[:][0]                       ----> Returns the 1st Column of the Matrix



--------------------->  4. Reshaping of Numpy Arrays



arr.reshape((3,3))                  ----> The arr 'array' should have 3*3 = 9 elements



--------------------->  5. Array Concatenation and Splitting



------> Concatenation



np.concatenate([arr1, arr2, arr3], axis = 0 / 1)        --->    We can concatenate as many as we want and along the row/column  
                                                                We can concatenate Mult-Dimen. Arrays as well



NOTE - Make sure the number of elements along the Rows/Columns are appropriate



Eg.
                                                                    
import numpy as np
print(np.concatenate([np.random.randint(1,10, (3,3)), np.random.randint(1,10, (3,3)), np.random.randint(1,10, (3,3))], axis = 0))

[[5 7 9]
 [2 4 3]
 [5 6 1]
 [4 2 2]
 [2 3 1]
 [4 2 4]
 [4 7 2]
 [2 4 2]
 [8 3 1]]



Eg.

import numpy as np
print(np.concatenate([np.random.randint(1,10, (3,3)), np.random.randint(1,10, (3,3)), np.random.randint(1,10, (3,3))], axis = 1))



[[2 5 9 3 9 3 1 9 7]
 [8 6 6 2 9 5 8 8 5]
 [9 8 1 6 1 6 4 8 3]]



------> Splitting (They Return n number of arrays)



For 1D Array :
    
    np.split(arr1, 3)                       ----> Splits the Array in 3 Equal Parts of  Array

    np.split(arr1, [3,7,9])                 ----> Splits the Array at the 3rd, 7th, 9th Indexes
                                            ----> [1st, 2nd, 3rd] [4th, 5th, 6th, 7th] [8th, 9th]



NOTE -  Look How it Actually Splits



For MultiDimensinal Array:
    
    np.split(arr1, 4, axis = 0 / 1)         ----> Splits the Matrix in 4 Equal Parts of Matrix
    
    np.split(arr1, [2,5], axis = 0)         ----> Splits the Matrix at the 2nd and 5th Rows Vertically
    
    np.split(arr1, [4,6,7,9], axis = 1)     ----> Splits the Matrix at the 4th, 6th, 7th and 9th Rows Vertically



------------------------------> Computation on NumPy Arrays: Universal Functions     <-----------------------------



Vectorization through ufuncs are more efficient than Python loops, especially as the arrays grow in size. 
For every loop in a Python script, you should consider whether it can be replaced with a vectorized expression.



NOTE - These UFuncs are Really Fast for Large Arrays



--------------------->  1. NumPys UFuncs



Ufuncs allow a NumPy user to remove the need to explicitly write slow Python loops



1.  Array arithmetic :

    Eg. 

    x = np.arange(4)

    x + 5

    x - 5

    x * 5

    x / 5                           -----> Divides every element of 'x' by 5, Similary every other operators works

    x // 5

    -x

    x ** 4

    x % 2

    -(0.5*x + 1) ** 2               -----> A combination of the above operators



2.  Absolute value :

    np.abs(x)                       ----> Returns absolute value of every element in 'x', works even for Complex Numbers
    
       

3.  Trigonometric functions:
    
    np.pi
    
    np.sin(x)                       ----> Returns the sin of every value in 'x' 
    np.cos(x)
    np.tan(x)
    
    np.arcsin(x)
    np.arccos(x)
    np.arctan(x)



4.  Exponents and logarithms


    np.exp(x)                       -----> Returns e^x for every element in 'x'
    
    np.sqrt(x)
    
    np.log(x)                       -----> log to the base 'e'
    np.log2()
    np.log10(x)



5.  Summing the Values in an Array

    np.sum(x)                       -----> Also Works for Multi-Dimensional Arrays
    
    

6.  Minimum and Maximum

    np.min(x)
    np.max(x)



7.  Multidimensional aggregates

import numpy as np

a = np.random.randint(3,10,(3,3))
print(a)
print(np.sum(a, axis = 0))

[[6 6 8]
 [3 8 5]
 [4 6 8]]

54


    np.sum(a, axis = 1)         ----->  Sames as every Ufunc but we can specify the axis as well



8. Other aggregation functions

Function Name     NaN-safe Version        Description

np.sum()          np.nansum()               Compute sum of elements
np.prod()         np.nanprod()              Compute product of elements


np.mean()         np.nanmean()              Compute median of elements
np.std()          np.nanstd()               Compute standard deviation
np.var()          np.nanvar()               Compute variance
np.median()       np.nanmedian()            Compute median of elements


np.min()          np.nanmin()               Find minimum value
np.max()          np.nanmax()               Find maximum value
np.argmin()       np.nanargmin()            Find index of minimum value
np.argmax()       np.nanargmax()            Find index of maximum value

np.percentile()   np.nanpercentile()        Compute rank-based statistics of elements

np.any(arr == 8, axis = 0/1)     N/A        Evaluate whether any elements are true
np.all(arr < 9,  axis = 0/1)     N/A        Evaluate whether all elements are true



9.  Comparison Operators as ufuncs



arr = np.array([1, 2, 3, 4, 5])

arr > 2         ----> Returns -->        array([False, False, True, True, True], dtype = bool)
                    Similarily for every other function

arr < 3

arr >= 1

arr <= 4

arr == 5

arr != 6 


----> Element by Element Comparison of 2 arrays


(2 * arr1) == (arr2 **2)



NOTE - All These functions work the same for MultiDimensional Arrays



------------------------------>   Conditioanl Selection in Numpy Arrays   <-----------------------------



import numpy as np
arr = np.arange(9).reshape(3,3)


print(arr)

[[0 1 2]
 [3 4 5]
 [6 7 8]]



NOTE - These are also called as 'Boolean Mask'



1. arr > 5

    print( arr > 5)

    [[False False False]
    [False False False]
    [ True  True  True]]



2. arr[arr > 5]

    print(arr[arr > 5])
    
    [6 7 8]

NOTE -  We can combine All Operator in Conditional Selection











------------------------->   NUMPY Indexing and Selection     <-------------------------



----------------------------------> For 1D Array  <------------------------


import numpy as np

arr = np.arange(3, 10)

We can slice the Array Just Like Slicing List in Python

arr[:3] = 100              # ---> This sets the value of arr[0] = arr[1] = arr[2] = 100 
                           ---> This is called Broadcast

NOTE - To avoid Memory Issues with very large arrays Numpy makes a reference to the Original Variable which changes its data too

slice_of_arr = arr[:]

slice_of_arr[:] = 12

print(arr)              #---> Both Will Give same output
print(slice_of_arr)

So to Avoid that we make a Copy of arr and then set it to slice_of_arr after which making any change to slice_of_arr will not affect arr

slice_of_arr = np.copy(arr) = arr.copy()
print(slice_of_arr)



---------------------------------> For 2D Array, matrix  <------------------------



import numpy as np

# 2 Ways to Select elements of a 2D Array

arr = np.array([[1,2,3,4],[12,13,543,123]])

1. arr[1][2]

2. arr[1,2]         ---> NOTE - This is available only in Numpy Arrays



------->  Conditional Selection



new_arr = arr[arr>5]      ---> This applies the condition on each element and returns the elements(if true) into a new array



---------------------------->  Numpy Operations  <----------------------



This is true only for Numpy Arrays --- This is also known as Broadcasting   

1. Array with Array

    arr +-*/ arr

2. Array with Scalar

    arr +-*/ some_number
arr ** some_number


NOTE - In Array with Array and Array with Scalar, the Length of Arrays should be same

3. Universal Array Functions (ufunc)

arr = np.sqrt(arr)
    = np.exp(arr)
    = np.max()
    = np.min()
    = np.sin(arr)         -----> i.e we can use trigonometric signs as well
    = np.log(arr)         -----> 
    = np.sum(arr)
    
    np.where('array_name' == np.NaN)                        
    np.delete('array_name', "index_of_element_to_be_deleted")



-------------------------------------------->   Python for Data Analysis - PANDAS     <--------------------------------------------------



arr = np.array(df1['column_name']) and then compute the data



Important Functions :


- Pandas is Open Source and built on top of NUMPY

Things to Learn :

1. Series & how they interact with Pandas
2. DataFrames of Pandas
3. Work of Missing Data
4. GroupBy of Pandas
5. Merging, Joining and Concatenating DataFrames of Pandas
6. Operations of Pandas
7. Data In and Out of Pandas, Such as CSV, Excel, SQL Files



-------------------------->  1. Series  <-------------------------



arr = np.array(df1['column_name']) and then compute the data



---------------------> Creating a Series



import numpy as np, pandas as pd 

list1 = ['a', 'b', 'c']
list2 = [10, 20, 30]
arr   = np.array(list2)
d     = {'a' : 10, 'b' : 20, 'c' : 30}



#Series can be made by 'LAD' - List, Arrays, Dictionary

print(pd.Series(d))         # We can create Series from Dictionary

print(pd.Series(d)['b':'c'])    NOTE -  We can used slicing on the Series made by Dicitionary and here we have
                                        [inclusive : inclusive]

Note - For Dicitionary the key are automatically sorted as per integers or alphabets



print(pd.Series(data = [sum, len, print], index = list2))           # We can also use builtin functions as data
                                                                    # Here list2 is the Label



x = pd.Series(data = list1, index = list2)      # Index can be a list of anything

print(x[10])        # Will Print 'a'



---------------------> (Adding Elements to a Series)   &   (Adding Series)



# lets say S1 and S2 are 2 Series Then (S1 + S2) will Add the Label/Indexs and return NAN if the data is not found in anyone 
# of them

S1 = pd.Series(data= list1, index = list2)
S2 = pd.Series(data= ['a', 'd', 'c'], index = [20,30,40])

S1['40'] = 'qwer'                   # These will add the respective elements
S2['50'] = 'asdf' 

print(S1+S2)

10    NaN
20     ba
30     cd
40    NaN



---------------------> Selecting Elements of a Series

import pandas as pd
S3 = pd.Series(data= ['a', 'd', 'c'], index = ['20','30','40'])

print(S3)

20    a
30    d
40    c

print(S3['30':'40'])

30    d
40    c

print(S3[0:])                       # Slicing can be done ONLY when all the indices are STRINGS

20    a
30    d
40    c



Builtin-Functions :
    
    1. x.values
        array([ 0.25, 0.5 , 0.75, 1. ])
    
    2. x.index
        RangeIndex(start=0, stop=4, step=1)             ---> Example
        
    3. x[1]   x[4:8]
        Here, Data can be accesssed by the associated index



------------------------> 2. DataFrames - 1  <--------------------------



NOTE -  In Series and DataFrames Integers are converted to Floats
        
        We can perform Set Operation on df1.index and df2.index

        We can change the values of a DataFrame by using ----> df1.loc['index_number', 'column_name'] = 'new_value'
        
        Eg. df1.loc[df1['some_columns_name'] == 'some_value', 'another_columns_name'] = 'some_new_value'

        We can make,        arr = np.array(df1['column_name']) and then compute the data



DataFrame acts in many ways 

1.Like a two-dimensional or structured array,

2.Like a dictionary of Series structures sharing the same index



Important Functions for DataFrame:
    1. read_csv()
    2. head()
    3. describe()
    4. memory_usage()
    5. astype()
    6. loc[:]
    7. to_datetime()
    8. value_counts()
    9. drop_duplicates(inplace = True, ignore_index = True)
    10. groupby()
    11. merge()
    12. sort_values(by='Name', inplace=True, ascending  = False)
    13. fillna(by = 38.5, inplace=True)
    14. reset_index( inplace = True)
    15. set_index(pd.Series([33,44,55,66]), inplace = True)
    16. nunique()
    17. index.names = ["Name1", "Name2"]
    
    
    
----------------------> Creating DataFrame



import numpy as np, pandas as pd
from numpy.random import randn



# NOTE - We can create DataFrame from Dictionary & List of List



pd.DataFrame(Elements, Row Indices, Column Indices)

np.random.seed(101)     ---> We set a seed to get the same random nums across different users   


To get INDEX & COLUMNS list:
    
    df1.index                   ---> Returns a List 
    df1.columns                 ---> Returns a List

WE can do anything just like the List



----------------------> Ways of Creating a DataFrame



Note - df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))   # Also a way of creating Multiple DataFrames



1. From a Single Series Object

A DataFrame is a collection of Series objects, and a single column DataFrame can be constructed from a single Series.

import pandas as pd
A = pd.DataFrame(data = pd.Series({'a' : 1, 'b' : 2, 'c' : 3}), columns = ['Letters'])
print(A)

   Letters
a        1
b        2
c        3



2. From a List of List



df1 = pd.DataFrame([[1,2,3],[4,5,6]])

print(df1)

   0  1  2
0  1  2  3
1  4  5  6

To get INDEX & COLUMNS list:
    
    df1.index                   ---> Returns a List
    df1.columns                 ---> Returns a List
    


3. Using a Dictionary & and a Dictionary of Series



print(pd.DataFrame({"A":[np.NAN, 2, 3], "B":[4, np.NaN, 5], "C":[6, 7, np.NaN],}, [96,97,98]))

    A    B    C
96  NaN  4.0  6.0
97  2.0  NaN  7.0
98  3.0  5.0  NaN



area = pd.Series({'California': 423967, 'Texas': 695662, 'New York': 141297, 'Florida': 170312, 'Illinois': 149995})

pop = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127, 'Florida': 19552860, 'Illinois': 12882135})

data = pd.DataFrame({'area':area, 'pop':pop})



print(data['area'])

California  423967
Florida     170312
Illinois    149995
New York    141297
Texas       695662



3. Using the Keywords , Elements, Row Indices, Column Indices



import numpy as np, pandas as pd
from numpy.random import randn

Using ---> randn

df = pd.DataFrame(randn(3,3), ['a','b','c'], [1,2,3])             # pd.DataFrame(Elements, Row Indices, Column Indices)
print(df)

    1         2         3
a  2.706850  0.628133  0.907969
b  0.503826  0.651118 -0.319318
c -0.848077  0.605965 -2.018168

print(df.head(2))                ----> Print n number of rows of the DataFrame

Using ---> randint



4. From a 2D Numpy array

B = pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])

        foo         bar
 a  0.865257    0.213169
 b  0.442759    0.108267
 c  0.047110    0.905718




------------------> Making Extra Columns & Rows in a DataFrame



-----> Adding Columns



df[4] = df[1] + df[2] | randn(3,1)    ---> This will add 1 more column 

df1['key'] = pd.Series([1,2,3])



-----> Adding Rows



1. df1.loc[len(df1.index)]  = [ 'Amy', 'King', 'Asdf']

2. df1.append({'Column1' : 'Amy' , 'Column2' : 'King', 'Column3' : 'Asdf'})



------------------> Deleting Rows and Columns



NOTE - df.shape == (3,3) which is a tuple so (axis = 0 == Rows) and (axis = 1 == Columns)

df.drop(['salary_in_usd'], axis = 1, inplace = True)  --> here inplace confirms that we want to delete the column permanently if inplace == False then it wont delete it permanently

df.drop(["b"], axis = 0, inplace = True)



------------------> Operations on DataFrames, Columns



df1 = df1 / df2

df1['column4'] = df1['column2'] / df2['column3']         ----> Can also do +, -, *, **

np.exp(df1)                     ----> Applies np.exp( ) on every element of the data frame and then returns a DataFrame

np.sin(df1 * (np.pi/4))         ----> Applies np.sin( ) on every element of the data frame and then returns a DataFrame



NOTE - Pandas will align indices in the process of performing the operation.

Eg . 

df1 =                   df2 = 
0   11                  2   14
1   22                  0   9
2   33                  1   28
                        3   84

df3 = df1 + df2

0   21
1   50
2   47   
3   NaN



----------------------------> 3. DataFrames - 2  <------------------------------



-----------> Selections / Conditinal Selections



(Here we can get and also give the data as well)



----> 3 Ways to retrive COLUMN Data : 



1. print(df[1])       ---> This is recommended --> Will print out a Series of Column 1

2. print(df.1)        ---> Will also print out a Series of Column 1

3. print(df[[1,2]])   ---> calling multiple Columns


NOTE -  Calling a Single Column will give a Series
        Calling Multiple Columns will give a DataFrame



----------> 3 Ways to retrive ROW Data :



1. df.loc[["a","c"], [2,3]]         ---> You can also get individual cell info

2. df.iloc[[1,2],[2,3]]             ---> Here we are using the Indices of the Rows and Columns

3. df.ix                            ---> This is a hybrid of 'loc' & 'iloc'



df.loc['ROW NAME', 'COLUMN NAME']
df.iloc['ROW INDEX', 'COLUMN INDEX']
df.ix[ 'ROW NAME' / 'ROW INDEX' , 'COLUMN NAME' / 'COLUMN INDEX' ]


Eg. Here 601,609 are row indices & job_title and remote_ratio are column names

1. df2.loc[601 , ['job_title','remote_ratio']]              

2. df2.loc[601:609 , 'job_title':'remote_ratio']

NOTE - CONTRARY TO USUAL PYTHON SLICES, BOTH THE START AND THE STOP ARE INCLUDED



---------> Applying a condition on every element of a DataFrame And  Getting True/False



import numpy as np, pandas as pd
from numpy.random import randn

df = pd.DataFrame(randn(4,4))

bool_df = df > 0      #i.e this returns a DF

print(df > 0)         ---> This will print the DF but with True/False at every cell 

       0      1      2      3
0  False  False  False  False
1  False   True   True  False
2  False   True  False   True
3   True  False  False   True




---------> Applying a condition on every element of a DataFrame And  Returning a DataFrame



print(df[ df > 0])    ---> This is same as print(df > 0) but for "false" cells it will write NAN

0         1         2         3
0       NaN  0.948022  0.641107  0.697116
1  0.769657       NaN  0.086328  0.282669
2  0.191764       NaN       NaN  0.087429
3  0.509276       NaN       NaN       NaN



---------> Applying a condition on a Column of a DataFrame



print(df[df[0]>0])          ---> This will return the DF but with column 0 elements > 0

          0         1         2         3
0  0.978765  0.634525  0.721986 -0.713260
1  0.958134 -1.006643 -0.465808 -1.786656

NOTE - when we use df[sdfasdf]  here "df" tells that it will return the whole DataFrame



------------> To add an index of [0,1,2,3.......n] besides the Original Index Column



----> df.reset_index(inplace=True)    ----> This is also a temporary change so we use inplace

index         0         1         2         3
0      0 -0.774455 -1.598083 -1.401992 -0.650946
1      1  1.019981  1.059463  0.022018  0.087523
2      2 -1.478124  0.219193 -0.678391  1.314480
3      3  0.208541  0.611723  0.377006  0.921247



------------> To add a Custom index besides the Original Index Column



----> df.set_index(pd.Series([33,44,55,66]),drop =  False, append = False, inplace = True)
print(df)

           0         1         2         3
33 -0.691260 -1.354530  1.075759  1.590082
44 -0.925570 -1.502910 -0.007169  0.788534
55  0.499383  0.204637 -0.553235 -1.241689
66  0.783800  1.097435 -0.860960  0.111078

Here, we can Set a New Index or  an Existing column as the Index column

df.set_index("salary_usd")



-------------------------------> 4. DataFrames - 3  <---------------------



NITE -  print(list(zip(["G1","G1","G1","G2","G2","G2",], [1,2,3,1,2,3],)))        ---> Gives out a List of Tuples

        [('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]

import numpy as np, pandas as pd
from numpy.random import randn

heir_index = list(zip(["G1","G1","G1","G2","G2","G2",], [1,2,3,1,2,3],))

heir_index = pd.MultiIndex.from_tuples(heir_index)

df = pd.DataFrame(randn(6,2), heir_index, ["A", "B"])   #---> Starting a MultiIndex Level DataFrame

print(df)
print(df.loc["G2"].loc[2]["A"])               #  ---> Gives out ( 2 G2 A ) as output

             A         B
G1 1 -0.538043  2.044674
   2 -0.283239  2.269420
   3 -0.036328  0.051043
G2 1  0.502767  2.788274
   2 -1.253619 -0.063398
   3  1.256906  0.100843
   
-1.2536188484529616                         ----> This is  the df.loc["G2"].loc[2]["A"] value

df.index.names = ["Group", "Numbers"]
print(df)

A         B
Group Numbers
G1    1       -0.982985  0.048941
      2        0.443154 -0.699686
      3       -0.106319  0.364710
G2    1        0.212434 -0.745369
      2       -0.659790  1.080584
      3       -0.176113  0.177451

print(df.xs("G1"))                   # ---> Returns the G1 Row

                A         B
Numbers                    
1       -0.286456 -0.417932
2        2.087379 -0.536791
3        0.064356  1.126378

print(df.xs(1, level ="Numbers"))   ---> Returns a Cross-Section of ROWS & COLUMNS

              A         B
Group                    
G1    -0.286456 -0.417932
G2     0.920734  0.569665

Same as .loc but it can grab all rows with 1 as their index
Here, we will get G1[1] & G2[1]



-------------------------------> 5. Missing Data



Refer to missing data in general as null, NaN, or NA values

Any Arithmatic Opertion on Nan will return NaN --------->   Eg. 1 + NaN = NaN, 0 * NaN = NaN

Pandas Treats --->  NaN = None



--------------->Functions used for Missing Data:
    
    
    
1. isnull()

    df2[df2['remote_ratio'].isnull()]       ---> Applies the condition on every element and returns the dataframe with, values in that 
                                                 column that are 'Null'
                                                 
    df2['remote_ratio'].isnull()            ---> Just Returns a Series with Values as True/False
    
    
2. notnull()

    df2[df2['remote_ratio'].notnull()]       ---> Applies the condition on every element and returns the dataframe with, values in that 
                                                 column that are 'Not Null'
                                                
    df2['remote_ratio'].notnull()            ---> Just Returns a Series with Values as True/False
    
                                                    
3. dropna()

4. fillna()



--------------> To Drop NaN Values in Rows/Columns



Pandas fills in the missing data as NULL or ANY Value

df.dropna( axis = 1 )                   ----->  Will Drop Column with at No.of NAN > 0               ---> axis = 0 by default

df.dropna( thresh=3 )                   ----->  Min. No. of Non-Null Values for the row to be kept
                                                i.e We need at least 3 Values for the row to kept

df.dropna(  how = 'any' / 'all')        ----->  Will Drop Row if 'Any Cell has a NaN Value' / 'All Cells have a NaN Value' 



------------> To Fill in Values of Nan in Rows/Columns



df.fillna(value = "New Vlue", inplace=True)

df["A"].fillna(value=df["A"].mean(), inplace=True)



-----------------------------> 6. Aggregation and GroupBy: Split, Apply, Combine



Aggregate functions :       NOTE : we can use, ----> axis = 'rows'/ 'columns'
    
    1.sum()
    
    2.mean(), median()
    
    3.min(), max()
    
    4.std(), var()              ----> Standard Deviation & Variance
    
    5.firt(), last()            ----> First & Last Item
    
    6.mad()                     ----> Mean Absolute Deviation
    
    7.prod()                    ----> Product of all items
    
    8.count()                   ----> Total Number of Items

    9.describe() :
        
        count 
        mean 
        std 
        min 
        25%             --->    25% of the Data
        50%             --->    50% of the Data
        75%             --->    75% of the Data
        max



Some Information on Statistics terms:
    
    1. Mode : The value that appears most often in a set of data values
    
        Eg. 
        
        You know that a college is offering 10 different courses for students. Now, out of these,
        the course that has the highest number of registrations from the students will be counted 
        as the mode of our given data (number of students taking each course)



    2. Mean : Average Value of the Data


    
    3. Median : In statistics and probability theory, the median is the value separating the higher half from the lower half of a data sample, 
                a population, or a probability distribution. For a data set, it may be thought of as "the middle" value



    4. Variance

    5. Standard Deviation

    6. Normal Deviation
    
    
    
GroupBy allows us to group together rows, based off of a column and perform an aggregate function on them(Sum, Standard Deviation, etc)

NOTE - To use groupby we need to give ---> index.names <----- to the MultiIndex



It Works on the Principle of    (SAC)      -----> SPLIT, APPLY, COMBINE


(Here, we are adding the numbers)

                        SPLIT   -------->   APPLY    -------->  COMBINE

A   43                  A   43              A   110             A   110
C   12                  A   67                                  B   129
B   36                                                          C   63
C   51                  B   36              B   129         
C   51                  B   93
A   67
B   93                  C   12              C   63
                        C   51



import numpy as np, pandas as pd
from numpy.random import randn

heir_index = list(zip(["G1", "G1", "G1", "G2", "G2", "G2", ], [1, 2, 3, 1, 2, 3],))

heir_index = pd.MultiIndex.from_tuples(heir_index)

df = pd.DataFrame(randn(6, 2), heir_index, ["A", "B"])

df.index.names = ["Groups", "Numbers"]
print(df)

                       A         B
Groups Numbers                    
G1     1        0.629621 -0.222161
       2        0.903285  1.503304
       3        0.044835 -1.483800
G2     1        0.009534  0.024455
       2        0.040723  1.148055
       3        0.937641  0.412544



-------------> Functions of Groupby



df.groupby()                            ---> This takes all the Groups
df.groupby("Groups")                    ---> This Gives a GroupBy Object
df.groupby("Groups").sum().loc["G2"]

print(df.groupby("Groups").sum().loc["G2"])
print(df.groupby("Groups").mean())              and etc.

NOTE - If there are strings in a Column then Pandas will ignore it automatically



df.groupby().describe().transpose()["G2"]       --->

.describe()     ---> Gives all the aggregate functions
.transpose()    ---> Makes the DataFrame/Matrix transpose
["G2"]          ---> Gives only the Info about G2



NOTE : The Groupby Function returns a 'DataFrameGroupBy' Object

We cannot print the content of the 'DataFrameGroupBy' Object, Instead we can only apply the Agg. Functions

With the GroupBy object, no computation is done until we call some 'AGGREGATE' on the object



-------------> Basic Groupby Operations



    1. df1.groupby['key1'].max()
     
            Here, this will 'groupby' the column 'key1' and then return the max() of all the columns in a 'DataFrame'
            
            The Groups will be the Row Names
       
    
    
    2. df1.groupby['key1']['data1'].max()
    
            Here, this will 'groupby' the column 'key1' and then return the max() of the column 'data1' in a 'Series'
            
            The Groups will be the Row Names
    
       

1. Column indexing 

    The GroupBy object supports column indexing in the same way as
    the DataFrame, and returns a modified GroupBy object.
    
    Eg.
    
    planets.groupby('method')['orbital_period'].median()
    
    Here,   'method'            == Column which is used to 'GroupBy'
            'orbital_period'    == Column/SeriesGroup along which we take the Median of the Data
            


2. Dispatch Methods

    Any method not explicitly implemented by the GroupBy object will be passed through and called on the groups,
    whether they are DataFrame or Series objects.

    Eg.
    
    planets.groupby('method')['year'].describe().unstack()
    
    Here, 
    
    describe() method is applied to each individual group, and the results are then combined within GroupBy and returned.
    
    Any valid DataFrame/Series method can be used on the corresponding GroupBy object,



-------------> Intermediate Groupby Operations          (TAAF)      (Transform, Aggregate, Apply, Filter )



df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],  'data1': range(6),  'data2': np.random.randint(0, 10, 6)})

key	data1	data2
0	A	0	7
1	B	1	5
2	C	2	8
3	A	3	7
4	B	4	5
5	C	5	4



1.  Transform







2.  Aggregate

    It can take a 'string', a 'function', or a 'list', and compute all the aggregates (i.e apply it on all the columns) at once
    
    1. Eg.
    
            df.groupby('key').aggregate(['min', np.median, max])
            
                data1	             data2
                min	median	max	     min  median	max
            key						
            A	0	1.5	    3	     3	    6.0	    9
            B	1	2.5	    4	     1	    2.5	    4
            C	2	3.5	    5	     5	    5.5	    6    



    NOTE - We can pass a dictionary mapping 'column names' to 'operations' to be applied on that column
        
    2. Eg.
    
    df.groupby('key').aggregate({'data1': 'min', 'data2': 'max'})
    

                        data1 data2
            key
            A           0       5
            B           1       7
            C           2       9

    Here, 'min' on 'data1' column & 'max' on 'data2' column


3. Apply



NOTE - 'x' is a DataFrame of group values


apply() method applies a function to the group elements. 

The function takes a DataFrame, and returns Pandas object (e.g., DataFrame, Series) or a Scalar


def norm_by_data2(x):                  # x is a DataFrame of group values

    x['data1'] /= x['data2'].sum()

    return x

df.groupby('key').apply(norm_by_data2)



4. Filter



NOTE - 'x' is a DataFrame of group values


Filter allows you to drop data based on the group properties. 

Eg. We  want to keep all groups in which the standard deviation is
larger than some critical value

def filter_func(x):                 # x is a DataFrame of group values
 return x['data2'].std() > 4


df.groupby('key').filter(filter_func)



-----------------------------------> 7. Concatenating, Merging and Joining  DataFrames



---------------------------> 1. Concatenating DataFrames    (pd.concat(......))



NOTE - Both of the DataFrames should have the same dimensions

import numpy as np, pandas as pd
from numpy.random import randn

df1 = pd.DataFrame(np.arange(1,10).reshape(3,3), index=np.arange(1, 4), columns = np.arange(1,4))
df2 = pd.DataFrame(np.arange(10,19).reshape(3,3), index=np.arange(1, 4), columns = np.arange(4,7))
df3 = pd.DataFrame(np.arange(19,28).reshape(3,3), index=np.arange(7, 10), columns = np.arange(7,10))



print(df1)

   1  2  3
1  1  2  3
2  4  5  6
3  7  8  9


print(df2)

    4   5   6
1  10  11  12
2  13  14  15
3  16  17  18


print(df3)

    7   8   9
7  19  20  21
8  22  23  24
9  25  26  27


-------------------------------------------------



print(pd.concat([df1, df2, df3], axis = 0))



     1    2    3     4     5     6     7     8     9
1  1.0  2.0  3.0   NaN   NaN   NaN   NaN   NaN   NaN
2  4.0  5.0  6.0   NaN   NaN   NaN   NaN   NaN   NaN
3  7.0  8.0  9.0   NaN   NaN   NaN   NaN   NaN   NaN
1  NaN  NaN  NaN  10.0  11.0  12.0   NaN   NaN   NaN
2  NaN  NaN  NaN  13.0  14.0  15.0   NaN   NaN   NaN
3  NaN  NaN  NaN  16.0  17.0  18.0   NaN   NaN   NaN
7  NaN  NaN  NaN   NaN   NaN   NaN  19.0  20.0  21.0
8  NaN  NaN  NaN   NaN   NaN   NaN  22.0  23.0  24.0
9  NaN  NaN  NaN   NaN   NaN   NaN  25.0  26.0  27.0



Note - While Concatenating along the rows it sees whether that column indices match up or not, if not then it creates the NaN matrix
       On sides accordingly

Eg. for df1 it doesnt have columns, 4,5,6,7,8,9 so it creates those columns with NaN values and then concatenates accordinly



------------------------------------------------



print(pd.concat([df1, df2, df3], axis = 1))



     1    2    3     4     5     6     7     8     9
1  1.0  2.0  3.0  10.0  11.0  12.0   NaN   NaN   NaN
2  4.0  5.0  6.0  13.0  14.0  15.0   NaN   NaN   NaN
3  7.0  8.0  9.0  16.0  17.0  18.0   NaN   NaN   NaN
7  NaN  NaN  NaN   NaN   NaN   NaN  19.0  20.0  21.0
8  NaN  NaN  NaN   NaN   NaN   NaN  22.0  23.0  24.0
9  NaN  NaN  NaN   NaN   NaN   NaN  25.0  26.0  27.0



Note - While Concatenating along the coumns it sees whether that row indices match up or not, if not then it creates the NaN matrix
       On top bottom sides accordingly

Eg. for df1 it doesnt have rows ,7,8,9 so it creates those columns with NaN values and then concatenates accordinly



-------------------------------------------



Note - We can concat them Vertically(axis = 0) or Horizontally(axis = 1)



-----------------------------> 2. Merging DataFrames    (pd.merge (....))



NOTE -  We use Merge when we have a Common Column. 
        We can have multiple Common Columns
        
        how = "inner", "right", "left", "outer"  & default is "inner"           ----> These are the different types of Joins
        on = ["key1", "key2"]                                                   ----> Name of the Columns to merge
        
import numpy as np, pandas as pd
from numpy.random import randn

df1 = pd.DataFrame(np.arange(1,10).reshape(3,3), index=np.arange(1, 4))
df1["key1"] = np.arange(1, 4).reshape(3,1)
df1.index.names = ['qwer']

df2 = pd.DataFrame(np.arange(10,19).reshape(3,3), index=np.arange(4, 7))
df2["key"] = np.arange(1, 4).reshape(3,1)

# When you are merging we are gonna merge it on a key column
print(df1)
print(df2)
print(pd.merge(df1, df2, how="outer", on=[0,1]))

   0_x  1_x  2_x  key  0_y  1_y  2_y
0    1    2    3    1   10   11   12
1    4    5    6    2   13   14   15
2    7    8    9    3   16   17   18



--------------------> Categories of Joins     



NOTE - These Concepts are used when both the Merging Columns have the 'Same number of Data' & 'Same Data'



1. One-to-one joins :

        df1 df2

        employee        group                   employee    hire_date
        0 Bob           Accounting          0   Lisa        2004
        1 Jake          Engineering         1   Bob         2008
        2 Lisa          Engineering         2   Jake        2012
        3 Sue           HR                  3   Sue         2014

        
        pd.merge(df1, df2)


                employee    group           hire_date
        0       Bob         Accounting      2008
        1       Jake        Engineering     2012
        2       Lisa        Engineering     2004
        3       Sue         HR              2014




2. Many-to-one joins :




3. Many-to-many joins :




--------------------> Some KeyWord Arguments of pd.merge()



NOTE - These Concepts are used when both the Merging Columns have the 'Same number of Data' & 'Same Data'



1. on = 'column_name'

    Tells the name of the columns on which the dataframes will be merged



2. left_on = 'left_column_name' , right_on = 'right_column_name' 

    Tells the name of the left column and the right column , and then the dataframes will be merged on these columns



3. left_index = 'left_index_name', right_index = 'right_index_name'

    Tells the name of the left index and the right index , and then the dataframes will be merged on these indexs



4. WE can combine '2.' and '3.' to form ---->   left_on = 'left_column_name' ,  right_index = 'right_index_name'



--------------------> SQL like joins in pd.merge()



This Concept is used when both the columns have some same data and all some different data

    how = "inner", "right", "left", "outer"  & default is "inner"           ----> These are the different types of Joins

1. inner

    Intersection of the two columns
    
4. outer

    Union of the two columns
    
2. right

    Concentrates on the right columns elements and discards the left column elements

3. left

    Concentrates on the left columns elements and discards the right column elements



-------------------> 'suffixes = ['_L', '_R']' Keyword



when some column names are common in both the dataFrames then the common column names are given a suffix 



---------------------------------> 3. Joining DataFrames        (df1.join(.....))
           
           
                        
Its a method for combining the columns of 2 "Different Indexed" DataFrames into 1 DataFrame                     
                        
Same as Merge but we use Index Columns instead of Keys Columns                      
                        
                        
df1.join(df2)                       
                        
           
                        
----------------------> 8. Operations                       
           
           
                        
import pandas as pd                     
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

print(df.head())            ---> returns all the rows of the DataFrame

df["col1"].unique()         ---> returns an array with the unique elements of "col1"

df["col1"].nunique()        ---> returns the number of unique elements    

df["col1"].value_counts()   ---> returns a table, how may times a value got repeated in the column   



NOTE - Apply Method is th Most Powerful Method in Pandas 

df["col1"].apply(func, axis = 0)                  ---> Applys the Custom Function along the axis
df.groupby('col1').apply(func, axis = 0)          ---> Applys the Custom Function on the groupby DataFrames


print(df.applymap(lambda x : x*2, na_action = False))      ---> Applys the Custom Function on the Every Single Element of the DataFrames
                                                                na_action --> To Ignore NaN or not


df["col1"].apply(len)       ---> Returns the Length of each element into a Series 

df.columns                  ---> Returns the name of the columns as an object

df.index                    ---> Returns the info about the Index

df.sort_values("col1")      ---> Sorts Values of the Column

df.isnull()                 ---> Returns a DF with Bool for Cell Items whether NAN or not



---------------------> Pivot Table 



Creating a MultiIndex Table out of the DataFrame

import pandas as pd                     
df = pd.DataFrame({"col2":[444,444,666,666], "col1":[1,2,1,2], "col3":['abc','def','ghi','xyz']})
print(df)

print(df.pivot_table(values = ["col3"], index=["col2"], columns=["col1"], aggfunc = "sum"))



---------------------> Vectorized String Operations



Handling and manipulating string data. This is a Very Important Concept for Real World DataSets.

Only Pandas has Vectorized String Operations And Numpy does not have it

For Array of 'Real Number', Numpy provides Vectorization of Operations But Not for 'Strings'



NOTE :  df1['method'].apply(len).max()
        df1['method'].str.len().max()

        These, both give out the same output



-------------> List of Pandas str methods that mirror Python string methods :



len()       lower()         translate()     islower()
ljust()     upper()         startswith()    isupper()
rjust()     find()          endswith()      isnumeric()
center()    rfind()         isalnum()       isdecimal()
zfill()     index()         isalpha()       split()
strip()     rindex()        isdigit()       rsplit()
rstrip()    capitalize()    isspace()       partition()
lstrip()    swapcase()      istitle()       rpartition()



NOTE -  df1['toys_name'].str.contains('string_name', re.IGNORECASE)         -----> Returns True if the string contains it
        df1['car_name'].str.cat('str1', 'str2')                          -----> Adds the two strings
        
        

How to  use these methods for Vectorized String Operations :
    
    Eg. 
    
    df1.str.len()       
    
    Applies this method to every element of the Series/DataFrame and returns that particular element for that particular cell
    
    This and every other method will return a Series/DataFrame depending on whether it is applied to a Series or a DataFrame
    


-------------> List of Methods Using Regular Expressions:
    
    
    
match()         Call re.match() on each element, returning a Boolean.
extract()       Call re.match() on each element, returning matched groups as strings.
findall()       Call re.findall() on each element.
replace()       Replace occurrences of pattern with some other string.
contains()      Call re.search() on each element, returning a Boolean.
count()         Count occurrences of pattern.
split()         Equivalent to str.split(), but accepts regexps.
rsplit()        Equivalent to str.rsplit(), but accepts regexps.


    Eg. 
    
    df1.str.match(r".................")
    
       

NOTE -  The ability to concisely apply regular expressions across Series or DataFrame entries
        opens up many possibilities for analysis and cleaning of data.



----->  Examples of Slicing and Split  



    Eg.
    
    df1.str[0:5]        -----> Slicing
    
    df1.str.split()     
    
    df1.str.split().str.get(-1)           -----> Applies split() & then get() to get the last element

    
    
---------------------> Working with Time Series



3 Types of Date and Time Data:  (SIPD)


    
    1. Time Stamps
    
        Particular moment in time (e.g., July 4th, 2015, at 7:00 a.m.).


    
    2. Time Interval & Period
    
        Time Interval   -   Length of time between a particular beginning and end point (eg.  the year 2015 )
        
        Period          -   Special case of 'Time Intervals' in which each interval is of uniform length and does not overlap 
                            (e.g., 24 hour-long periods constituting days).   
                            

                            
    3. Time Delta / Duration

        An exact length of time (e.g., a duration of 22.56 seconds).









































---------------------> High-Performance Pandas: eval() and query()



For More Efficiency, Less Memory Use, For Less Computational time and Better Syntax we will use eval() and query()


1.  For example, consider the following expression :
    
    mask = (x > 0.5) & (y < 0.5)

    Because NumPy evaluates each subexpression, this is roughly equivalent to the following:
        
        tmp1 = (x > 0.5)
        tmp2 = (y < 0.5)
        mask = tmp1 & tmp2
        


2.  Consider the following expression :

    x = df[(df.A < 0.5) & (df.B < 0.5)]
    
    This is roughly equivalent to this :
        
        
    tmp1 = df.A < 0.5
    tmp2 = df.B < 0.5
    tmp3 = tmp1 & tmp2
    x = df[tmp3]



Explanation ---->   Every intermediate step is explicitly allocated in memory. 
                    If the x and y arrays are very large, this can lead to significant memory and computational overhead
                    
                    The ability to compute this type of compound expression element by element, 
                    without the need to allocate full intermediate arrays is very EFFECIENT
                    
                    Same Concept applies to DataFrames
                    


-------------------> pandas.eval() for Efficient Operations             (eval = Evaluate)


df1, df2, df3, df4 = (pd.DataFrame(numpy.random.rand(nrows, ncols)) for i in range(4))



2 ways to do it the 'SUM' of all the DF's :


    1. df5 = df1 + df2 + df3 + df4
    
    This is in-effecient way to do it   
    

    2.  We can compute the same result via 'pd.eval' by constructing the expression as a 'STRING'

    df5 = pd.eval('df1 + df2 + df3 + df4')

        This expression is about 50% faster (and uses much less memory)
        


--------->  Operations supported by pd.eval()



1.  Arithmetic operators

    pd.eval('-df1 * df2 / (df3 + df4) - df5')

2.  Comparison operators

    pd.eval('(df1 < df2) & (df2 <= df3) & (df3 != df4)')

3.  Bitwise operators

    pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')

4.  Object attributes and indices

    pd.eval('df2.T[0] + df3.iloc[1]')
    


---------> Column-Wise Operations for DataFrames



1. Benefit of the eval() method is that columns can be referred by 'Name'


df1.eval('(A + B) / (C - 1)')                   ---->   Here A, B, C are column names



2. Assignment in DataFrame.eval()

df1.eval('D = (A + B) / C', inplace=True)       ---->   Here, D is the new columm which is being Added

        

3. Local Variables in DataFrame.eval()

df.eval('A + @column_mean')                     ---->   Here, we use the '@' operator to add the 'column_mean' int 
                                                        to all the elements of column 'A'



----------------------------> DataFrame.query() Method

pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')      ---->   This cannot be expressed using the 'df1.eval()' syntax,



For Conditional Selection we can use the 'df1.query()' method :

    df1.query('A < 0.5 and B < 0.5')


We can also use Local Variables :

    df1.query('A < @some_int and B > @another_int')






-------------------- -> 9. Data Input and output



CSV 
EXCEL
HTML 
SQL
                 
import pandas as pd

pd.rea





------------------------------------> Matplotlib by Derek Banas <----------------------------------------



---------------------------> Functional Plots



--------> Simple Plot



#%%

import matplotlib.pyplot as plt, numpy as np, pandas as pd

%matplotlib inline                      
# ---> Allows us to see the plot inside the Jupyter Notebook

x = np.linspace(0, 20, 100)

plt.xlabel("THE X LABEL")
plt.ylabel("The Y Label")
plt.title("The Title")
plt.plot(x,x**2)

#%%


--------> Multiple Plot

#%%

plt.subplot(1,2,1)
plt.plot(x,x**3,'r')

plt.subplot(1,2,2)
plt.plot(x,x**3,'b')


#%%



---------------------------> Object Oriented Plots



---------> Using Figure Objects


# Figure is an object that contains all the plot elements and it can contain many axes

#%%
fig1 = plt.figure(1,(4,2),100,'b','r')
axes1 = fig1.add_axes([0.1,0.1,0.9,0.9])
axes1.set_xlabel("XLABEL")
axes1.set_ylabel("YLABEL")
axes1.set_title("TITLE")
axes1.plot(x,x**2,label = "x - x^4")





#%%







------------------------------------>  Python for Data Visualization - Matplotlib <----------------------------------------



---------------------> Matplotlib Part - 1 <----------------------------



Its the most popular plotting library for Python

import matplotlib.pyplot as plt, numpy as np 

%matplotlib inline              ---> Allows us to see the plot inside the Jupyter Notebook

plt.show()                      ---> Draws the Plot in VSCode

x = np.linspace(0, 5, 110)
y = x ** 2


 
----------------> 2 ways of Creating Matplotlibb Plots



1. Functional Method              ---> This is simple way of plotting

%matplotlib inline                ---> Allows us to see the plot in the jupyter notebook

#%%
import matplotlib.pyplot as plt, numpy as np 
x = np.linspace(0, 5, 110)
y = x ** 2


plt.xlabel("X Label")       
plt.ylabel("Y Label")
plt.title("Title")
plt.plot(x,y,'r-')                      #   ---> Giving some attributes to the plot
plt.show()                              #  ---> Use this in .py file to view the plot

plt.subplot(1,2,1)                      #  ---> plt.subplot(No. of Rows, No. of Columns, Plot No. we are refering to)
plt.plot(x,y,"r")


plt.subplot(1,2,2)                      #  ---> plt.subplot(No. of Rows, No. of Columns, Plot No. we are refering to)
plt.plot(y,x,"b")


#%%
2. Object Oriented Method           (This is a Better Way of Creating a MatplotLib Plot)

We make Figure Objects and call Methods off of it

#%%

fig = plt.figure()                       #   ---> Think of it as an Imaginary Canvas and we can add a Set of Axis

x = np.linspace(0, 5, 110)
y = x ** 2
plot2 = fig.add_axes( [2, 2, 2, 2] )     # .add_axes(Left, Bottom, Width, Height), 0 to 1, The percent of that blank canvas we want, 
                                         # Here we are actually placing the Bottom-Left Corner of the Plot
plot2.set_xlabel("THE X LABLE")
plot2.set_ylabel("THE Y LABLE")
plot2.set_title("THE TITLE")

plot2.plot(x,y)

#%%
fig = plt.figure()
axes1 = fig.add_axes([0,0.1,2,2])
axes2 = fig.add_axes([0.3,0.4,1,1])

axes1.plot(x, x**9)
axes2.plot(x,x**2)

NOTE -  1. Here, We added a Figure
        2. Added Axes to the Figures
        3. Then plotted our Data on it i.e ( x,y )

#%%


#%%
---------------------------> Matplotlib Part - 2 <----------------------------





#%%
import matplotlib.pyplot as plt, numpy as np
fig, axes = plt.subplots(3,3)       # Here plt.sublplots() automatically calls fig_addaxes( ) for us
plt.tight_layout()                  # Corrects the Layout as more spaced out
x = np.linspace(0, 10, 100)
axes.plot(x,x**2)
axes.plot(x,x**2)



#%%



------------------------------------>  Python for Data Visualization - Seaborn <----------------------------------------



------------------------------------>  Python for Data Visualization - Pandas Builtin Data Visualization <----------------------------------------



------------------------------------>  Python for Data Visualization - Plotly & Cufflinks <----------------------------------------



------------------------------------>  Python for Data Visualization - Geographical Plotting <----------------------------------------



------------------------------------>  Introduction to Machine Learning <----------------------------------------



Domain Knowledge Plays a Very Important Role in Machine Learning



---------------------> Supervised Learning Algorithms



Trained Labeled examples, i.e An Input where the desired OutPut is known

Eg. 

1. Spam vs Legitimate Emails       ------>  Someone went through the emails and we know the spams and the legitimate one
                                            so the program looks at the previous data and predicts the new one as spam/legitimate
                                            
2. Positive vs Negative Movie Reviews

The Algo get Inputs and Outputs, learns from it and then compare its outputs to the 'real outputs' to find error.
It then modifies the model.


Machine Learning Process :
    
                                            -----------> Test Data ------------>-----
                                           |                                         |
    Data Acquisition    ---> DataCleaning  ---> Model Training and Building ---> Model Testing ---> Model Deployment
                                                    |                                |
                                                    -<-- Adjust Model Parameters <---|                                            



1. We Acquire the data
2. Clean that Data and format it so that the machine learning model can accept it
3. Split the Data into Training Set, Validation Set, Test Set
4. Iterate and Tune the Parameters of the Model until its ready to deploy



To make the model more accurate we split the data into 3 sets

1. Training Data    ---> Used to Train Model Parameters.

2. Validation Data  ---> Used to Know which Parameters to Adjust.

3. Test Data        ---> Used to get final Parameter Metric, this is the data the model has never seen before.
                         This is how the model will work in the real world and we are not allowed to make anymore changes.
                         It will be the true performance of the model on UNSEEN DATA.



---------------------> Evaluating performance - Classification Error Metrics



Binary Classification - eg. Predict if the Image is a Dog/Cat

For Supervised Learning we FIT/TRAIN the model on TRAINING DATA and then TEST the model on TESTING Data. 
And then we compare the models prediction to the actual values

We organize the PREDICTED VALUES vs REAL VALUES in a CONFUSION MATRIX



------------> Classication Metric to Judge our Model (ARPF):
    
    
    
1. Accuracy :   <--------------------------------------------
    
    
    
    Accuracy ==         Total Number of Correct Predictions
                    --------------------------------------------
                            Total Number of Predictions
                            
                            

    Well Balanced Target Classes --->   Number of Images of Dog  ~= Number of Images of Cats
                                        51 Dog Images ~= 49 Cat images
                                        
    Un - Balanced Target Classes --->   Number of Images of Dog  >>> Number of Images of Cats
                                        99 Dog Images >>> 1 Cat images
                                        
    In the UnBalanced Target Class we will get 99% Accuracy which wont work in the actual world



2. Recall (Identification of 'Malignant Tumor') (Here we give only the 'Malignant Tumor' Images)  <-------------------------------------

    

    What Proportion of Actual Positive(Ground Truth) was identified correctly?
    
    Recall =                   True Positive                   =                     Correct Identification
                --------------------------------------------         -------------------------------------------------------
                       True Positive + False Negative                     Correct Identification + Wrong Identification
                       
                       
            Correct Identification = True Positive
            Wrong Identification   = False Negative


    Is it really able to recall what the 'Malignant Tumor' looks like?
    
    Eg. For a 'Malignant Tumor' identification if Recall = 0.11 then we can say that 
        the Model Correctly identifies 11% of all 'Malignant Tumor' 
        
        NOTE -  We give all the 'Malignant Tumor' images and then the model identifies if it is a 'Malignant Tumor'
                So, either it will say it is 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is TRUE POSITIVE
                or it will say it is not 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is FALSE NEGATIVE
                
                
                
3. Precision (Picking out 'Malignant Tumor' out of all the Tumors) (Here we give the Images of all Tumors)  <-------------------------------------------



    What proportion of positive identifications was actually correct?

    Precision =            True Positive                   =               Correct Answer
                ---------------------------------------         --------------------------------------
                    True Positive + False Positive                   Correct Answer + Wrong Answer
                       
                       
            Correct Answer = True Positive
            Wrong Answer   = False Positive


    Is it able to pick 'Malignant Tumor' out of all the 'Tumor' Images
    
    Eg. If the Precision for picking 'Malignant Tumor' = 0.4 then we can say that the model correctly picks 'Malignant Tumor' 40% of the Time

    NOTE -  We give the images of all 'Tumors' and the model picks 'Malignant Tumor' out of them so either it will be 'Malignant Tumor' or NOT
            So, either it will be say it is 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is TRUE POSITIVE
            or it will say its is 'Malignant Tumor' and in realtiy is not 'Malignant Tumor', then it is FALSE POSITIVE



------------------------------------------------------------------------



NOTE -  There is a Tug of War between RECALL and PRECISION i.e
        If we increase Precision , then Recall Decreases
        & If we increase Recall, then Precision Decreases



------------------------------------------------------------------------



4. F1-Score   <------------------------------------------



    F1 Score is the Harmonic Mean of Recall and Precision


    
    Harmonic Mean =      2 * A * B                 
                    --------------------      
                           A + B



    F1 =      2 * Recall * Precision                
         --------------------------------       
                Recall + Precision


                
    We use Harmonic Mean cause it helps us deal with extreme values
    
    Eg. for Precision  = 1.0 and Recall =  0.0
    
    we have Harmonic Mean == 0 and Average  == 0.5


--------------> Confusion matrix

                            Prediction Positive         Prediction Negative

Condition Positive          True  Positive              False Negative

Condition Negative          False Positive              True  Negative



---------------------> Evaluating performance - Regression Error Metrics
 


Regression is a Task when a Model trys to predict continuous Values
(Classification is prediction of Categorical Values) 

Eg. Predict the price of a house given its features is a 'Regression Task'



----------------->  Evaluation Metric for Regression (MMR) :
    
    
    
1. Mean Absolute Error



Summation (i = 1, n)  |y(i) - y^(i)|
                    ------------------
                            n


y(i)            --->    Actual Value
y^(i) [y cap]   --->    Predicted Value by the model
n               --->    No. of Elements in the data


Mean Absolute Error does not punish large error

NOTE - Here Large Errors can create trouble so we have to use 'MEAN SQUARED ERROR'



2. Mean Squared Error



It is the 'Mean' of the 'Squared Error'



Summation (i = 1, n)  ( |y(i) - y^(i)| )**2
                    -----------------------
                               n


Here Large Error are more noted than MAE

NOTE : That Squaring of the Errors cause the squaring of the Actual Values as well

        Eg. Squaring of the Dollars in the House Price prediction which is difficult to interpret



3. Root Mean Square Error
    
    
        (    Summation (i = 1, n)  ( |y(i) - y^(i)| )**2      )
   SQRT (                        --------------------------   )
        (                                    n                )


y(i)            --->    Actual Value
y^(i) [y cap]   --->    Predicted Value by the model
n               --->    No. of Elements in the data



NOTE :  It Punishes the Large Error Values and has the same values as y

        To Get an Intuition of the Model Performance, Compare your 'ERROR METRIC' to the 'AVERAGE VALUE' in your data



---------------------> Machine Learning with Python



----------> Scikit learn



Every model in ScikitLEarn is used via an 'Estimator'

General Form of importing a Model :
    
    from sklearn.'family' import 'Model'
    
Eg. from sklearn.linear_model import LinearRegression

    Here, 'linear_model'     is a family of models  
          'LinearRegression' is an Estimator Object (i.e it has parameters which can be set when instantiated)
           Estimator Object  is the Model itself


Estimator Parameters : Params can be set when instantiated and have defult Values

After creating the model with our parameters, we then fit it on a training model

NOTE -  Remember to split the data in 'TRAINING SET' & 'TEST SET' 

#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X,y = np.arange(10).reshape(5,2), range(5)

print(X) 
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, shuffle=True)
print(y_test)
print(y_train)
print(X_test)
print(X_train)

#%%



NOTE - 'sklearn.cross_validation' has been deprecated and the new code is 'sklearn.model_selection'

1. We fit/train our model on a training data by 'model.fit(X_train, y_train   )' method
2. We predict values                         by 'model.predict(X_test)'          method
3. We then compare the values with the "TEST DATA", here the evaluation method depends on the ALGORITHM used
   eg. Regression, Classification, Clustering, etc



For All Estimators :
    model.fit()         : Fit training data
    model.fit(X, y)     : For Supervised Learning
    model.fit(X)        : For Un-Supervised Learning, since they are unlabeled data

    

For Supervised Estimators :
    model.predict(X_new)    : For a trained model, it predicts the label for each object in the array
    model.predict_proba()   : For Classification Problems some estimators provide this method.
                              It return the Probability for each LABEL, label with highest probability is returned by model.predict()
    model.score()           : Return value b/w 0-1, For Classication & Regression, larger value mean greater fit              



For Un-Supervised Learning :
    model.predict()         : Predicts Label For Clustering Algorithms
    model.transform()       : Transform new data into new basis
    model.fit_tranform()    : Efficiently performs fit and transform on the same input data



------------------------------------>  Linear Regression <----------------------------------------
    








