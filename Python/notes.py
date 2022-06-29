  Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell



"""

"""

      Python

--> num=12
    name = Rohit 
    'my number is {} and name is{}'.format(num,name)
    
--> s = 'hello'
    s[0] will give 'h'
    
--> Slicing     s[includes this : doesnt not includes this]

    s[0:] will give 'hello'
    s[:3] will give 'hell'
    
--> List
 
    my_list = [1,2,3,4]   
    my_list.append(5)  --> [1,2,3,4,5]
    my_list[0]  -> 1
    
    List can also be Sliced

    nesting --> nest = [1, 2, [3,4,'target']]
    nest[2][2][3]  -> 'g'
    
    
    --> LIST Comprehension
       
    out = [num**2 for num in my_list]   
    
-->  Tuples are immutable

    my_tuple = (1,2,3)

--> Dictionaries (Key Value Pairs)

    d = {'key1':'value', 'key2': 123}
    
    d['key1'] == 'value'
    
    
    We can put anything in place of the Value like list, Dictionaries or 
    even any variables
    
--> Set 

    s = {1,2,3}         contains ony unique elements
    
    set([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3])
    {1,2,3}

--> == , !=, and, or

--> if and elif

    if 1<2:
        print("hello")
    elif 2>4:
        print("laugh")
    else :
        print("jump")
    
    
--> for loop

    seq = [1,2,3,4,5]

    for item in seq:
        print(item)
            
--> while loop 

    i = 1
    
    while i < 5:
        print('i is {}'.format(i))
        i = i + 1;
        
--> range()

    range(2,8)
    --> 2,3,4,5,6,7
    
    for x  in range(10):
        print(x)

--> Functions

    def my_func(param1 = "Default Name"):
        print(param1)

    my_func()           --> will print 'Default Name'
    my_func("Rohit")    --> will print "Rohit"


        def my_func(sum):
            return(sum**2)

        my_func(6)



--> Pypi and pip install

--> Modules and Packages
 
Modules are .py scripts and Packages are a collection of Modules
 
add __init__.py file in your folder to make it a Packages
 
if __name__ == "__main__":
    #tells if the .py file is called directly or imported
    true == directly
    false == imported

--> Error Handling

3 Keywords:
    1) try - block of code to be executed, may lead to Error
    2) except - block of code to be executed, if there is error in try block
    3) finally - block of code to be executed, regardless of error
    4) else - with no error it will execute
    
    while TRUE:
        try:
            "ssdaf"
        except TypeError:     // can also look for specific errors
            "sdfas"
            continue
        else:
            "sadfsdf"
            break
        finally:
            "sadsdf"
    
-> Pylint and Unittest (Testing tools)

    Pylint - lib that looks at the code and reports possible issues
    
    Unittest - built-in lib , allows to test code and check desired output
    
    Python has a set of style convention rules known as PEP-8
    
    import unittest as ut


--> Python decorators (Advanced topic)

    decorators allow us to "decorate" a Function
    
    def new_decorator(og_func):
        
        def wrap_func():
            
            print("dssdv")
            
            og_func()
            
            print("weqwwq")
            
        return wrap_func
    
    @new_decorator()  // this is will input func_needs_decorator in new_decorator()
    def func_needs_decorator():
        print("sdfsdf")
        
    -> function inside a function
    
    def cool(args):
        return qewr
        
        def supercool():
            return 100
    
    
 --> Python Generators(Advanced topic) (YIELD, NEXT, ITER)
 
    Generator functions send back a single value and then pick up where
    it left 
    
    Generates a sequence over time and the main difference will be 
    the 'yield' statement 
    
    Gen. func. will automatically suspend and resume their execution from 
    the last point of value generation 
    
    Advantage is that it computes one value at a time instead of computing 
    entire series of value up front.
    
    NOTE - range is a gen. function
    
    list(range(0,10))    --> Creates a list of 10 elements in Memory
 
    def create_cubes(n):
        for x in range(n):
            yield x**3
    
    for num in create_cubes(10):
        print(num)        
    
    FIBONACCI NUMBERS:
        
    def fib_num(n):
        a = 1
        b = 1
        for x in range(n):
            yield a
            a, b = b, a+b
            
    
    NEXT FUNCTION:
        
    def simple_gen():
        for i in range(5):
            yield i
            
    print(next(simple_gen()))
    
    ITER FUNCTION: Allows us to iter upon an object or a data type
    
    s = iter('hello')
    print(next(s))
        

    
 
 

