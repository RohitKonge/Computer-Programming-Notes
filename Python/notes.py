  Jupyter Notebook

--> Shift + Enter = Run Current Cell and Add new Cell

--> Alt + Enter = Add New Cell



"""

"""

      Python
      
      
      type(    )  --> give info about the data type of the object()
      pass        --> passes the function or class without doing anything 
      del         --> deletes the object from the memory


--> num=12
    name = Rohit 
    'my number is {} and name is{}'.format(num,name)
   f'my number is {num} and name is {name}'
    
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
    
    my_set = [1,2,3]   --> this is a list
    my_set = set()    --> This will turn it into a set
    
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

    Methods  - Fucntions that are defined in a Class are called Methods

    def hello():
        print("asdas")
    
    hello()     --- > Will EXECUTE the function
    hello       --- > Will GIVE INFO about the function and can be used 
                      as an argument

    def my_func(param1 = "Default Name"):
        print(param1)

    my_func()           --> will print 'Default Name'
    my_func("Rohit")    --> will print "Rohit"


        def my_func(sum):
            return(sum**2)

        my_func(6)


--> Class (Methods, Attributes/Characteristics, Instance)

class Dog():      -->Class Name is given in CamelCase

    species = 'mammal'    ---> CLASS OBJECT ATTRIBUTE 
                          ---> SAME FOR AMY INSTANCE OF THE CLASS
                          ---> Can be called as Dog.species (this is convention) or self.species
        
    def __init__(self, breed, name, age = 12):    --> Also known as the constructor in C++ and is called every time an instance is created
                                                  --> 'self' connects the instance to the Class
        self.breed = breed                        --> 'breed' is the Attribute of the Dog Class
        self.name = name
        print(breed)
        
    def bark(self, num):                --> Methods for the Class
        print(num + self.species)       --> 'num' is not connected to the particular instance of the class 
                                            so self is not used 

sample_instance = Dog('Labra', 'asdf', 26) # --> This is an object and an instance of the 'Sample' Class
sample_instance.breed
sample_instance.name

sample_instance.species
    
--> INHERITANCE AND POLYMORPHISM AND ABSTRACT CLASSES

INHERITANCE

class Dessert():
    
    def __init__(self):
        print("sweet dish")
    
    def what_kind(self,kind):
        self.kind = kind
        print(self.kind)
    
class IceCream(Dessert):
    
    NOTE --> Desert.__init__(self) executes only once 
             Dessert().__init__(self) executes twice
    
    def __init__(self, name):
        Dessert.__init__(self)
        self.name = name
        
    def give_name(self):
        print(self.name)

vanilla = IceCream('chocolate vanilla')
vanilla.what_kind('sweet kind of vanilla')
vanilla.give_name()    


POLYMORPHISM

    (POLYMORPHISM will be used much later in your python carreer)
    
    Polymorphism means the condition of occuring in several different forms

class Dessert():
    
    def __init__(self):
        print("sweet dish")
    
    def what_kind(self,kind):
        self.kind = kind
        print(self.kind + "Good Desert")

class Soup():
    
    def __init__(self):
        print("nice soup")
    
    def what_kind(self,kind):
        self.kind = kind
        print(self.kind + "Good Soup")
        

food1 = Dessert()
food1.what_kind("Vanilla")

food2 = Soup()
food2.what_kind('Carrot')

def our_food(food3):
    print(food3.what_kind("jhjkhk"))

our_food(food1)
our_food(food2)


ABSTRACT CLASSES

We NEVER EXPECT to create an instance of the abstract class and it is designed to only 
serve as a BASE CLASS

class Animal():
    
    def __init__(self,name):
        self.name = name
        
    def speak(self):
        raise NotImplementedError("Subclass must implement this abstract method")
    
---> It expects us to use the abstract class to override the 'speak' method
    
my_Animal = Animal('fred')

class Dog(Animal) :
    def __init__(self, name, age):
      self.name = name
      self.age = age

    def speak(self):        --> This has been based on the Abstract Class - Animal
        print("Woof woof")



--> SPECIAL/MAGIC/DUNDER Methods of a CLASS





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


--> Python DECORATORS (Advanced topic) - 

    Used when we want to add new capabilities to our function
    
    
    --> DECORATOR and RETURN A FUNCTION
    
    def new_decorator(og_func):
        
        def wrap_func():
            
            print("dssdv")
            
            og_func()
            
            print("weqwwq")
            
        return wrap_func    // this will execute the wrap_func() function
    
    @new_decorator  // this will input func_needs_decorator in new_decorator()
    def func_needs_decorator():
       
        print("sdfsdf")
        
    NOTE --> We can call func_needs_decorator() and it will execute new_decorator
             with func_needs_decorator() as an argument
             
             
  
    --> function inside a function
    
    def cool(args):
        
        def supercool():
            return 100
    
        return supercool
    
  
  
    --> Passing a function as an argument
    
    def other(some_def_func):
        
        print("weerwer")
        
        return(some_def_func())
    
    
    
 --> Python GENERATORS (Advanced topic) (YIELD, NEXT, ITER)
 
    Generator functions send back a single value and then pick up where
    it left 
    
    Generates a sequence over time and the main difference will be 
    the 'yield' statement 
    
    Gen. func. will automatically suspend and resume their execution from 
    the last point of value generation 
    
    Advantage is that it computes one value at a time instead of computing 
    entire series of values up front.
    
    NOTE - range is a gen. function
    
    list(range(0,10))    
 
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
    
    ITER FUNCTION: Allows us to iter upon ANY  OBJECT  or a DATA TYPE
    
    s = iter('hello')
    print(next(s))
        

    
 
 

