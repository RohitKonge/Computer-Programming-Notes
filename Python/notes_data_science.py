  Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell


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

