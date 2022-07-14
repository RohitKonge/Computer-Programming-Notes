Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell

NOTE - Defining a Lot of variables takes a lot of memory so as we get better at Python we move more and more towards One Liner Code
NOTE - and , or are used for   Single Bool comaprisions 
       &   , |  are used for a Series Bool Comparisions


---------------------------- PYTHON FOR DATA ANALYSIS - NUMPY - --------------------------

Its a Linear Algebra Library

np.NaN == Not A Number

Aggregate Function == A function that takes in lots of individual values and then returns a single value

---------------------    NUMPY ARRAYs     -------------------------

import numpy as np 

Numpy Arrays differ from Python List because of its ability to Broadcast

Eg. 

a = np.array([1,2,3,4])
b = np.array([5])
print(a*b)                              -----> Broadcast

we can call randint as ---> from numpy.random import randint

NOTE - All elements in the array will be FLOAT64

arr = np.array([[1,2,3],[4,6,9]])      # ---> Here different length list are Deprecated

print(arr)

print(np.arange( 0, 10, 3))    # (Inclusive, Exclusive, Steps to Jump) Just like Python Range
#  Returns an array

np.arange(50)                   ---> Returns a list from 0 to 49

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
print(arr.reshape(3,3).shape)     ----> This is way to make a 1D Array to N dimensional Array

print(arr.dtype)                ---> Returns the data type of the elements


---------------------    NUMPY Indexing and Selection     -------------------------


------------> For 1D Array

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

------------> For 2D Array, matrix

import numpy as np

# 2 Ways to Select elements of a 2D Array

arr = np.array([[1,2,3,4],[12,13,543,123]])

1. arr[1][2]

2. arr[1,2]         ---> NOTE - This is available only in Numpy Arrays

------->  Conditional Selection

new_arr = arr[arr>5]      ---> This applies the condition on each element and returns the elements(if true) into a new array



---------------------     Numpy Operations     -------------------------

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

import numpy as np, pandas as pd 

list1 = ['a', 'b', 'c']
list2 = [10, 20, 30]
arr   = np.array(list2)
d     = {'a' : 10, 'b' : 20, 'c' : 30}

print(pd.Series(d))         # We can create Series from Dictionary

print(pd.Series(data= [sum, len, print], index = list2))        # We can also use builtin functions as data
                                                                # Here list2 is the Label

x = pd.Series(data= list1, index = list2)

print(x[10])        # Will Print 'a'

# lets say S1 and S2 are 2 Series Then (S1 + S2) will give Add the Label/Indexs and return NAN if the data is not found in both of them

S1 = pd.Series(data= list1, index = list2)
S2 = pd.Series(data= ['a', 'd', 'c'], index = [20,30,40])

print(S1+S2)

10    NaN
20     ba
30     cd
40    NaN

---------------> 2. DataFrames - 1

NOTE -  In Series and DataFrames Integers are converted to Floats

---> Creating DataFrame

import numpy as np, pandas as pd
from numpy.random import randn

# NOTE - We can create DataFrame from Dictionary & List of List

print(pd.DataFrame([[1,2,3],[4,5,6]]))

   0  1  2
0  1  2  3
1  4  5  6

print(pd.DataFrame({"A":[np.NAN, 2, 3], "B":[4, np.NaN, 5], "C":[6, 7, np.NaN],}, [96,97,98]))

    A    B    C
96  NaN  4.0  6.0
97  2.0  NaN  7.0
98  3.0  5.0  NaN


np.random.seed(101)     ---> We set a seed to get the same random nums across different users   

pd.DataFrame(Elements, Row Indices, Column Indices)

import numpy as np, pandas as pd
from numpy.random import randn

df = pd.DataFrame(randn(3,3), ['a','b','c'], [1,2,3])
print(df)

    1         2         3
a  2.706850  0.628133  0.907969
b  0.503826  0.651118 -0.319318
c -0.848077  0.605965 -2.018168

print(df.head(2))                ----> Print n number of rows of the DataFrame

---> Getting Data From DataFrame (Here we can get and also give the data as well)

3 Ways to retrive COLUMN Data

1. print(df[1])       ---> This is recommended --> Will print out a Series of Column 1

2. print(df.1)        ---> Will also print out a Series of Column 1

3. print(df[[1,2]])   ---> calling multiple Columns


NOTE -  Calling a Single Column will give a Series
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

print(df[ df > 0])    ---> This is same as print(df > 0) but for "false" cells it will write NAN

0         1         2         3
0       NaN  0.948022  0.641107  0.697116
1  0.769657       NaN  0.086328  0.282669
2  0.191764       NaN       NaN  0.087429
3  0.509276       NaN       NaN       NaN

print(df[df[0]>0])          ---> This will return the DF but with column 0 elements > 0

          0         1         2         3
0  0.978765  0.634525  0.721986 -0.713260
1  0.958134 -1.006643 -0.465808 -1.786656

NOTE - when we use df[sdfasdf]  here "df" tells that it will return the whole DataFrame

---> To add an index of [0,1,2,3.......n] besides the rows

df.reset_index(inplace=True)    ----> This is also a temporary change so we use inplace

index         0         1         2         3
0      0 -0.774455 -1.598083 -1.401992 -0.650946
1      1  1.019981  1.059463  0.022018  0.087523
2      2 -1.478124  0.219193 -0.678391  1.314480
3      3  0.208541  0.611723  0.377006  0.921247


df.set_index(pd.Series([33,44,55,66]),drop =  False, append = False, inplace = True)
print(df)

           0         1         2         3
33 -0.691260 -1.354530  1.075759  1.590082
44 -0.925570 -1.502910 -0.007169  0.788534
55  0.499383  0.204637 -0.553235 -1.241689
66  0.783800  1.097435 -0.860960  0.111078


---------------------> 4. DataFrames - 3

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


---------------------> 5. Missing Data


-----> To Drop Nan Value in Rows/Columns

Pandas fills in the missing data as NULL or ANY Value

df.dropna(axis=1)    -----> Will Drop Column with at least 1 NAN
df.dropna(thresh=2)  -----> Will Drop if no. of non-Nan values are less than 2


-----> To Fill in Values of Nan in Rows/Columns

df.fillna(value = "New Vlue", inplace=True)

df["A"].fillna(value=df["A"].mean(), inplace=True)

-------------------- -> 6. GroupBy

GroupBy allows us to group together rows, based off of a column and perform an aggregate function on them(Sum, Standard Deviation, etc)

NOTE - To use groupby we need to give index.names to the MultiIndex 

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

df.groupby()                            ---> This takes all the Groups
df.groupby("Groups")                    ---> This Gives a GroupBy Object
df.groupby("Groups").sum().loc["G2"]

print(df.groupby("Groups").sum().loc["G2"])
print(df.groupby("Groups").mean())              and etc.

NOTE - If there are strings in a Column then Pandas will ignore it automatically

Also, Here .sum(), .std(), .count(), max() are Aggregate functions

df.groupby().describe().transpose()["G2"]       --->

.describe()     ---> Gives all the aggregate functions
.transpose()    ---> Makes the DataFrame/Matrix transpose
["G2"]          ---> Gives only the Info about G2



-------------------- -> 7. Concatenating, Merging and Joining  DataFrames


---------> Concatenating DataFrames

NOTE - Both of the DataFrames should have the same dimensions

import numpy as np, pandas as pd
from numpy.random import randn

df1 = pd.DataFrame(np.arange(1,10).reshape(3,3), index=np.arange(1, 4))
df2 = pd.DataFrame(np.arange(10,19).reshape(3,3), index=np.arange(4, 7))
df3 = pd.DataFrame(np.arange(19,28).reshape(3,3), index=np.arange(7, 10))

# print(df1)
# print(df2)
# print(df3)

print(pd.concat([df1, df2, df3], axis = 1))

     0    1    2     0     1     2     0     1     2
1  1.0  2.0  3.0   NaN   NaN   NaN   NaN   NaN   NaN
2  4.0  5.0  6.0   NaN   NaN   NaN   NaN   NaN   NaN
3  7.0  8.0  9.0   NaN   NaN   NaN   NaN   NaN   NaN
4  NaN  NaN  NaN  10.0  11.0  12.0   NaN   NaN   NaN
5  NaN  NaN  NaN  13.0  14.0  15.0   NaN   NaN   NaN
6  NaN  NaN  NaN  16.0  17.0  18.0   NaN   NaN   NaN
7  NaN  NaN  NaN   NaN   NaN   NaN  19.0  20.0  21.0
8  NaN  NaN  NaN   NaN   NaN   NaN  22.0  23.0  24.0
9  NaN  NaN  NaN   NaN   NaN   NaN  25.0  26.0  27.0

Here the Indices do not contain an element so we get NaN


print(pd.concat([df1, df2, df3], axis = 0))

    0   1   2
1   1   2   3
2   4   5   6
3   7   8   9
4  10  11  12
5  13  14  15
6  16  17  18
7  19  20  21
8  22  23  24
9  25  26  27

Note - We can concat them Vertically(axis = 0) or Horizontally(axis = 1)

---------> Merging DataFrames

NOTE -  We use Merge when we have a Common Column. 
        We can have multiple Common Columns
        
        how = "inner", "right", "left"  & default is "inner"
        on = ["key1", "key2"]       
        
import numpy as np, pandas as pd
from numpy.random import randn

df1 = pd.DataFrame(np.arange(1,10).reshape(3,3), index=np.arange(1, 4))
df1["key"] = np.arange(1, 4).reshape(3,1)
df2 = pd.DataFrame(np.arange(10,19).reshape(3,3), index=np.arange(4, 7))
df2["key"] = np.arange(1, 4).reshape(3,1)

# When you are merging we are gonna merge it on a key column

print(pd.merge(df1, df2, how="inner", on="key"))

   0_x  1_x  2_x  key  0_y  1_y  2_y
0    1    2    3    1   10   11   12
1    4    5    6    2   13   14   15
2    7    8    9    3   16   17   18

print(pd.merge(df1, df2, how="inner", on=["key1", key2]))
Use this if we have mutiple key columns


---------> Joining DataFrames
                        
Its a method for combining the columns of 2 "Different Indexed" DataFrames into 1 DataFrame                     
                        
Same as Merge but we use Index Columns instead of Keys Columns                      
                        
                        
df1.join(df2)                       
                        
                        
-------------------- -> 8. Operations                       
                        
import pandas as pd                     
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

print(df.head())            ---> returns all the rows of the DataFrame

df["col1"].unique()         ---> returns an array with the unique elements of "col1"

df["col1"].nunique()        ---> returns the number of unique elements    

df["col1"].value_counts()   ---> returns a table, how may times a value got repeated in the column   


NOTE - Apply Method is th Most Powerful Method in Pandas 

df["col1"].apply(func)      ---> Applys the Custom Function on the elements

print(df.applymap(lambda x : x*2))


df["col1"].apply(len)       ---> Returns the Length of each element into a Series 

df.columns                  ---> Returns the name of the columns as an object

df.index                    ---> Returns the info about the Index

df.sort_values("col1")      ---> Sorts Values of the Column

df.isnull()                 ---> Returns a DF with Bool for Cell Items whether NAN or not

--------------> Pivot Table 

Creating a MultiIndex Table out of the DataFrame

import pandas as pd                     
df = pd.DataFrame({"col2":[444,444,666,666], "col1":[1,2,1,2], "col3":['abc','def','ghi','xyz']})
print(df)

print(df.pivot_table(values = ["col3"], index=["col2"], columns=["col1"], aggfunc = "sum"))


-------------------- -> 9. Data Input and output

CSV 
EXCEL
HTML 
SQL
                 
import pandas as pd

pd.rea






















