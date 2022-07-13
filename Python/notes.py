    Python
      
      
    type(    )          --> give info about the data type of the object()
    pass                --> passes the function or class without doing anything 
    del                 --> deletes the object from the memory
    map(func, iter1)    --> Just like a "for i in range(0,10)" this applies iter1 elements to the Function. Usually we use Lambda Functions.
    filter(func, iter1) --> Here we return a Boolean in the Lambda Expression & Return/Filter the Elements from the Iterable. 
    
    NOTE - Defining a Lot of variables takes a lot of memory so as we get better at Python we move more and more towards One Liner Code
    NOTE - and , or are used for   Single Bool comaprisions 
           &   , |  are used for a Series Bool Comparisions
    
    
    list(map(lambda arguments : expression , sequ))
eg. list(map(lambda x:x**3, [1,2,3,4]))

    
    list(filter(lambda arguments : expression , sequ))
eg. list(filter(lambda x:x**3 >= 8, [1,2,3,4]))


print(list(map(lambda x:x**3, [1,2,3,4])))          --> Use of lambda Expression


NOTE - Always use the backward slash when typing the Path of file/folder/image .etc 

Eg. Python/roosevelt.jpg            ----> This is correct
    Python\roosevelt.jpg            ----> This is wrong



NOTE - If you get an error as " object is not Callable"

Eg. img.filename()          ---> Gives an error as  -> 'str' object is not callable
    img.filename            ---> Then remove the parentheses , this will work

NOTE - If you get an error as "Unexpected Indent"

    print("asdasf")         ---> This will give the error "Unexpected Indent"
    
print("asdasf")             ---> This is the correct Indentation

--> num=12
    name = Rohit 
    'my number is {} and name is{}'.format(num,name)
   f'my number is {num} and name is {name}'
    
--> s = 'hello'
    s[0] will give 'h'
    
--> Slicing     s[includes this : doesnt not includes this]

    s[0:] will give 'hello'
    s[:3] will give 'hell'
    s[:]  will give 'hello'
    
--> List
 
    my_list = [1,2,3,4]   
    my_list = []          # Creates an empty list
    my_list.append(5)  --> [1,2,3,4,5]
    my_list[0]  -> 1
    
    List can also be Sliced

      for i in my_list[1:]:
          This will iterate from index 1 to the end


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
    
    Tuple Unpacking:
        
    x =[(1,2),(3,4)(4,5)]
        
    for a,b in x:
        print(a,b)
    
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
    
    One such use case is when we want a "open" method to open PDF, Excel, Word, CSV we can 
    use Polymorphism and have muliple methods with the name 'open'

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

Here we override the builtin functions to suit our Data Type made with the use of Class

class Book:
    def __init__(self, name, page):
        self.name = name
        self.page = page

    def __str__(self):
       return self.name
   
    def __len__(self):
        do something
        
    def __del__(self):
        do something

print(str(Book("Potter", 240)))  --> we overrided the str for our class to return string representation of the BOOK Class

likewise we can also override ---> len, del 



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
        

-->Advanced Python Packages and Modules


--> from collections import Counter

print(Counter('sadfsdfsdfsdf'))  --> Returns a dictionary 
Counter.values()
Counter.most_common(self)


--> from collections import defaultdict

This sets a default 'value' for the new 'key's which are created

d = defaultdict(lambda : 10)

d['correct'] = 100

print(d['WRONG KEY'])

now DEFAULTDICT is a little different from DICT here if we search a key that is
not present in the dict then it will return '10' as its value


--> from collections import namedtuple

NAMEDTUPLE expand over TUPLE by having 'Named Indicies'

namedtuple has a numeric connection as well as Named index connection

Dog = namedtuple('Dog1', ['name', 'age'])

my_dog = Dog("sam", 12)


--> Opening and Reading, Files and Folders using Python OS Modules

Python OS Module and Shell Utilities Module allow us to move or delete Files

f = open('practice.txt', 'w')
f.write('This is a practice file')

import os   --> Get current/all files in a dir and works across all Operating Sys.

print(os.getcwd())  --> cwd = Current Working Dir
print(os.listdir())

os.walk(top) --> Walks through every Folder, Sub-Folder, and File in its path

NOTE - Os module permanently deletes files so we import 'send2trash' to send it to Trash Can

import send2trash

send2trash.send2trash('e:\Computer Programming\GithubCPNotes\practice.txt' )

import shutil      --> Helps to Move the files around
                   --> Shell Utilities Module
                   
shutil.move('practice.txt', 'e:\Computer Programming\GithubCPNotes')


--> Datetime Module

import datetime

datetime

-->Python Math and Random Modules

Random Module contains a lot of mathematical random functions and functions 
for grabbing a random item from a python list


import math   --> just learn Numpy library, this is a basic math module
math.cos(x)
import random

random.randint(0,100)  --> Gives a random number from 0 to 100
random.seed(101)       --> Sets a sequence for the infinite numbers

my_list = list(range(0,20))

random.choice(my_list)      --> This chooses 1 element from the list

random.choices(population = my_list, k = 10) --> Chooses 10 random numbers from the list but it can repeat values
--> k is number of items we want

random.sample(population = my_list, k = 10) --> Sames as random.choices but will choose unique number everyt.isnumeric()

random.shuffle(my_list)

random.uniform(a, b)


 --> Python Debugger
 
import pdb
 
pdb.set_trace()


-----------------------------> Python Regular Expression - 1

- to search a small string in a large string use

    a = "dog" in "who is that dog"

    The Problem here is that we need to know the exact pattern and structure



- Now we want to search for a GENERAL TYPE of data like all emails in a file

    Regular Expressions(RegEx) allow us to search general patterns in text data

    Eg. user@email.com          --> Here we dont know "user" and "email" but we do know '@' and '.com'

    Phone Number Pattern -  (555)-555-5555
    Regex Pattern        - r"(\d\d\d)-\d\d\d-\d\d\d\d"
                         = r"(\d{3})-\d{3}-\d{4}")
                         
    Here '\d'            - Identifier 
        '\d{3}'          - {3} is called Quantifier
        (  ) , ' - '     - Format String

import re 

txt = "asd qwe etry"
pattern = "asd"

 -- re.search(pattern, txt)     --> returns None if it is not present
                                Return span(x,y) i.e the indexs of the founded pattern

NOTE - This returns a "match" Objects where we can use the following Methods

- match.span()
- match.start()
- match.end()


 -- re.findall(pattern, txt)    --> Returns a List with the number of findings in the txt
 
 -- To Search & Iterate through the txt we Use
 
 for match in re.finditer(pattern, txt):
    print(match)
    
    
-----------------------------> Python Regular Expression - 2


---- Character Identifiers 


            Description         Pattern Code            Match
\d          Digit               file_\d\d\d             file_123
\w          Alphanumeric        \w-\w\w                 A-b_
\s          Whitespace          a\sb\sc                 a b c

\D          Non-Digit           \D\D\D                  ABC
\W          Non-Alphanumeric    \W\W\W\W                *-+=)
\S          Non-Whitespace      \S\S\S\S                YoYo


---- Quantifiers


            Description                         Pattern Code        Match
            
?           Once or None                        plurals?            plural
*           Occurs Zero or More Times           A*B*C*              AAAACC
+           Occurs One  or More Times           \d-\d+              9-99999 

{3}         Occurs Exactly 3    Times           \d{3}               897
{3,5}       Occurs 3 to 5       Times           \d{3,5}             1234
{3,}        Occurs min     3    Times           \d{3,}              1234567

import re

f = re.findall(r"\d\d\d{3,6}", "   33342323232344121   11111111111111111  2222222222222")
print(f)

-- if we want to search for multiple patterns in the text we can use re.compile(pattern)


f = re.fullmatch()(re.compile(r"(\d{1})(\d{2})(\d{3})"), "33342323232344121   11111111111111111  2222222222222")
print(f.group(3))



-----------------------------> Python Regular Expression - 3

---- Additional Regex Syntax

import re


---> Finding multiple strings

re.search(r"cat | dog", "asd cat adsfasd f asdfds sda fdsa dog")        ---> Returns cat or dog Match objects


---> Finding the string and n letters before the string 
print(re.findall(r"....at", "sd sd swe waat s at adsf asdat erw tat"))  ---> Return at + the four letters before that note- this can also inlcude whitespace 


---> Start and End 

print(re.findall(r"^\d", "1dsf  sdsdf 4s  fs 75asdfasd"))      #--> ^ this symbol searchs at the start of the string

print(re.findall(r"\d$", "asd3 asda asaq 999asda wqeqw12"))    --> $ this symbol searchs at the end of the string


Exclude something from the search and Search everything else


re.findall(r"[^\d]+", string)

re.findall(r"[^?.!]+", string)

" ".join(clean)     ---> It joins all the individual strings in to a 1 string



Include something from the search and Search everything else

import re

print(re.findall(r"[\w]+-[\w]", "qwe asd-qwe wqefvxb-wqexcv ewqr"))

['asd-q', 'wqefvxb-w']



txt1 = "catfish"
txt2 = "catnap"
txt3 = "caterpillar"

print(re.search(r"cat(fish|nap|terpillar)", txt1))



-----------------------------> Timing your Python code

To check which Solution for a question is the fastest we use this

import timeit

3 ways of doing it 

1. Tracking the time elapsed 

import time 

start_time  = time.time()
result      = func(1000000)             ---> this function gets repeated 
end_time    = time.time()

elapsed_time = end_time - start_time


2. Timeit Module                ---> The most efficient Way

import timeit

print(timeit.timeit(stmt = "func_one(100)", setup = "def func_one(n) : return[str(num) for num in range(n)]", number=100000))

here,   stmt(Statement) = code we actually wanna test                       = Function with arguments
        setup           = anything that need to be set beforehand           = Function definition
        number          = Number of times the function is to be repeated

stmt and setup are passed as strings


3. %%timeit    which works only on Jupyter Notebook

%%timeit
func_one(100)



-----------------------------> Unzipping and Zipping Files















----------------------- WebScrapping with Python  --------------------------


--> Intro to Web Scraping

Automating the gathering of data from a website is called WEB SCRAPING 

DATA includes images or information 

3 important Things to keep in Mind:

1. Rules of web Scraping
    - Always get permission 
    - Too many scraping requests will block your IP Address
    - Sometimes sites automatically block scraping software

2. Limitation of Web Scraping
    - Every Website is unique so we Need unique Web Scraping Scripts for everyone 
    - Slight Change/Update to a website will break your Web Scraping Code

3. Basic HTML & CSS 


--> Setting up web Scraping

pip install requests  ---  Allows us to make a request to a website and then grab the info off of it. 

pip install lxml      --- Used by BeautifulSoup to decipher whats inside the requests

pip install bs4       --- BeautifulSoup


--> Grabbing the Title

import requests

result = requests.get("https://en.wikipedia.org/wiki/Jonas_Salk")

print(type(result))

result.text  # --> this gives the html as a giant string

import bs4

soup = bs4.BeautifulSoup(result.text,"lxml")

a = soup.select(".toctext")  # soup.select("p")  

 print(a[0].getText())

for item in a:
    print(item.getText())


--> Grabbing All Elements of a Class


A big part of web scraping is knowing what string syntax to pass in to the soup.select(" ") method

            Syntax                                   Match results
            
1.  soup.select("div")                     --> Elements with "div" Tag

2.  soup.select("#some_id")                --> Elements with id="some_id"

3.  soup.select(".some_class")             --> Elements with class="some_class"

4.  soup.select("div example")             --> Elements named "example" with div element

5.  soup.select("div > example")           --> Elements named "example" directly within div element,
                                               wit nothing in between

(The Code for grabbing Class is written in "Grabbing the title" section)



--> Grabbing an Image 

Images have their own Link ending with .jpg or .png

Here we serach for the <img> tag 

import requests

result = requests.get("https://en.wikipedia.org/wiki/Jonas_Salk")

import bs4

soup = bs4.BeautifulSoup(result.text, "lxml")

imglist = soup.select(".thumbimage")

 print(imglist[0]["src"])

imglink = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Roosevelt_OConnor.jpg/220px-Roosevelt_OConnor.jpg")

 print(imglink.content)

f = open("roosevelt.jpg", "wb")   -->  wb = write binary

f.write(imglink.content)

f.close()


--> Working with Multiple Pages and Items 


We want to grab multiple elements, most likely across multiple pages.

to go across multiple pages we can use .format(" ") on the url

Eg. 

http://books.toscrape.com/catalogue/page-2.html

site_page = (http://books.toscrape.com/catalogue/page-{ }.html).format(" ")

And then Loop over the pages to get the relevant info 

import requests
import bs4
        
for j  in range(1,51):
    result = requests.get(("http://books.toscrape.com/catalogue/page-{}.html").format(j))

    soup = bs4.BeautifulSoup(result.text, "lxml")

    two_star_list =  soup.select(".product_pod")
    
    for i in two_star_list:
        if(i.select(".star-rating.Two")):
            print(i.select("a")[1]["title"])
    
    
Note - When accessing nested elements make sure to use [0] or [1] .select gives a list

Note -  i.select("asd").text     will print the string




----------------------- Working with Images with Python  --------------------------


We will use PILLOW to work with images

Then we open, save and interact with Images



Pillow is a fork of the PIL - Python Imaging Library which has easy to use function calls 

Sometimes working with Large Images we can get 'IOPub data rate exceeded' error

The default data rate maybe too low for some of the images we are Working

Close the Browser and on CMD write---

jupyter notebook --NoteBookApp.iopub_data_rate_limit = 1.0e10



from PIL import Image

img = Image.open("Python/roosevelt.jpg")

img.show()
img.size()
img.format_description()
img.filename

--> Cropping Images

Here the co-ordinates start from the Top-Left corner and they are all co-ordinates from the origin(i.e (0,0)) and not lengths 

print(img.size)

NOTE -  we are passing the numbers as a TUPLE

img.crop((0,0,100,100)).show()    --> This is a rectange with (0,0) as the top-left corner and 100 as x-co-ordiante and 100 as y-co-ordiante

img.paste(im=img.crop((0,0,50,50)),box = (0,0), mask = img.crop((0,0,50,50)) ) --> paste the im(image) , mask helps to view the images which is below

img.resize((300,400))       --> Changes the Size of the Image 

im.rotate(90)               ---> Rotates the Image by 90 degree


--> Color tranparency

img.putalpha(Value)    ---> Changes the Transpanrency, 0 <= Value <= 255

img.save("purple.png")    ---> This saves the image and if it already exists , it will override it or make a new Image 





----------------------- Working with PDFs and CSV files in Python  --------------------------


----> Working with CSV Files

- top sentence contains the names of the columns

- We can export EXCEL, GOOGE SPREADSHEETS to .csv files but it only exports the info 

- .csv files do not contain images and macros

- "Openpyxl" is designed specifically for EXCEL Files & "python-excel.org" has other Excel based python libs

- "Google Sheets Python API"

     - Python interface for working with Google Spreadsheets
     - Make changes to spreadsheets hosted online 
     
     

Step for a typical CSV files

1. Open the file
2. CSV. Reader
3. Reformat it into a Python Object . List of Lists 


import csv

data = open("example.csv")

csv_data = csv.reader(data)       -- Converting it into CSV Data

data_line = list(csv_data)        -- Reformatting

- Now this will give a UnicodeDecode Error 

- Encoding is ability to read/not-read the different types of special characters 
    Eg. '@' symbol

data = open("Python/example.csv",mode= "rt", encoding='utf-8')       
---> This helps it to read the special characters


csv_data = csv.reader(data)     

data_line = list(csv_data)

-- On getting UnicodeDecode Error look for different ENCODING online and it is useful if we know what kind of special characters are in our CSV file

print(data_line[1])
print(len(data_line))

all_emails = []  

for i in data_line[1:3]:
    all_emails.append(i[3])   



--> Writing to a CSV File

import csv

file_to_output = open("Python/to_save_file.csv", mode="w", newline = "")

csv_writer = csv.writer(file_to_output, delimiter = ",")    ---> Delimiter is something that separates the columns, in this case since its a CSV(comma sepraterd values) file we have ","

- we can have ";" or "\t" , TSV(tab separated values)
 
csv_writer.writerow([["1", "2", "3"],["4", "5", "6"]])   ---> Here we have a list of list 
csv_writer.writerow(["sdsdafsdf", ["4", "5", "6"]])
     
file_to_output.close()
     
     
     
     
--> Working with PDF Files in Python


- We will use PYPDF2 library

Images, Table make a PDF unreadable by python although there are PAID PDF programs that can extract from these files

import PyPDF2

f = open("Python/Working_Business_Proposal.pdf", "rb")

pdf_reader = PyPDF2.PdfFileReader(f)

print(pdf_reader.numPages)

page_one = pdf_reader.getPage(0)

page_one_text = page_one.extract_text

print(page_one_text)
     
f.close()     ----> The Above Code is for Reading only





f = open("Python/Working_Business_Proposal.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(f)

first_page = pdf_reader.getPage(0)

pdf_writer =  PyPDF2.PdfFileWriter( )
     
pdf_writer = pdf_writer.addPage(first_page)

pdf_output = open("Python/SomePdf.pdf","wb")

pdf_writer.write(pdf_output)
f.close
     
     
     
     
---------------------- Emails with Python  --------------------------
    

-----> Steps to send Emails with Python

1. Connecting to an Email Server
2. Confirming Connection
3. Setting a Protocol
4. Logging on
5. Sending Message
     
SMTP (Simple Mail Transfer Protocol)

- Here we use an APP PASSWORD instead of our NORMAL PASSWORD to let Gmail know that I am the one trying to access my account

import smtplib

smtp = smtplib.SMTP("smtp.gmail.com", 587)

smtp.ehlo()   --> This methods calls and makes the connection to the server
              -->   NOTE- This method call should be made directly after creating the SMTP Object

smtp.starttls()

import getpass

email    = getpass.getpass("Enter Email : ")
password = getpass.getpass("Enter Password : ")
smtp.login(email, password)

--> Set up app password on Google 

from_addr = email
to_addrs = email
subject = input("Enter Subject: ")
msg = input("Enter Message: ")

smtp.sendmail(from_addr, to_addrs, msg)

--- If we get an empty dictionary that means the email was successful 

smtp.quit()    ----  this will close the connection
     
     
     
-----> Steps for Recieved Email with Python

import imaplib

imap = imaplib.IMAP4_SSL("imap.gmail.com")

import getpass
email = getpass.getpass("Email : ")
password = getpass.getpass("Password : ")

imap.login(email, password)

imap.login("rohitkonge08@gmail.com", "thepasswordisincorrect")


---------------------- Advanced Python Objects and Data Structures --------------------------

-----> Advanced Numbers

hex(number)     --> Hexadecimal form of a Number  -->   0x200  = 512
bin(number)     --> Binary      form of a Number  -->   0b1011 = 11

2**4        == 16
pow(2,4)    == 16
pow(x,y,z)  == (x^y) % z     <----

abs(-3)     == 3

round(3.1)  == 3.0          --> Round will always a Int Number
round(3.9)  == 4.0

Python also has a Math library


-----> Advanced Strings

s = "hello world"

"h" in s        ---> Will return True 

s.capitalize()  ---> Capitalizes the first letter
s.upper()       ---> Capitalizes the whole word 
s.lower()       ---> Lowers the whole word
s.count("o")    ---> Counts the total number of 'o'
s.find("o")     ---> Finds the 1st index of the occurence of 'o'

s.isalnum()     ---> These are usful when using
s.isalpha()         for NLP(Natural Language Processing)

s.islower()     ---> True if all letter are lowercase 
s.isspace()     ---> True if all characters are whitespace 
s.endswith("o") ---> True if it ends with 'o'

s[-1]           ---> Will give the last letter
s[-2]           ---> Will give the 2nd last letter

s.split()       ---> Split the String with Whitespace
s.split("o")    ---> Returns a list with the string being separated at every 'o'

s.partition("ll") ---> Returns a list with the input, the part before and after it 
                       i.e it will have 3 elements in the list
                       
\t              ---> tab 

NOTE - The difference between split and partition - In SPLI the input does not get returned in the List but in PARTITION we do get the input in the list.



-----> Advanced Sets

s = set()
s.add(1)
s.add(2)

print(s)            ---> Prints the whole Set

s.clear()           ---> Removes all elements of the Set

s_new = s.copy()    ---> Returns a copy of s and changes to the Original Set wont affect the new Set

s_new.add(3)
s_new.add(4)

s.difference(s_new)             ---> Returns new set by deleting all the common elements

s.difference_update(s_new)      ---> Removes the elements that are in both s and s_new and return s

s.discard(2)                    ---> Removes 2 from s , wont show error if 2 is not present in s

s.intersection(s_new)           ---> Returns a new set which has elements common in s and s_new

s.intersection_update(s_new)    ---> Returns s as a new set which has elements common in s and s_new 

s.isdisjoint(s_new)             ---> Returns True if s intersection s_new == Null

s.issubset(s_new)               ---> Returns true if all elements of s are in s_new

s.issuperset(s_new)             ---> Returns true if all elements of s_new are in s
 
s.symmetric_difference(s_new)   ---> Returns all the elements that are in exactly one of the sets

s.union(s_new)                  ---> Returns a set that has the combination of the 2 sets

s.update(s_new)                 ---> Returns s as a new set which have union of elements of s_new



-----> Advanced Dictionaries

d = {"k1" : 1, "k2" : 2}

-- Dictionary Comprehension  

a = {x:y for x,y in zip([1,2],[3,4])}

NOTE - zip methods yields tuples and stops when the shortest input range is exhausted



-- Iterating through Dictionaries

for k in a.items() / a.keys() / a.values() :
    print(k)


-----> Advanced Lists

l = [1,2,3]

l.count(3)              ---> Returns the Times number 3 is present

l.append([4,5,6])       ---> Adds [4,5,6] to the list ->  [1,2,3,[4,5,6,]]
l.extend([4,5,6])       ---> Adds  4,5,6  to the list ->  [1,2,3,4,5,6]

l.index(2)              ---> Returns the index of 2

l.insert(3,[4,5])       ---> Inserts [4,5] before index 3

l.pop()                 ---> By default it removes the last element of the list 
l.pop(2)                ---> Removes the element at index 2 in the list

l.remove(2)             ---> Removes the first occurence of the value '2' in the list

l.reverse()             ---> Reverses the list

l.sort()                ---> Sorts the list


---------------------- Introduction to GUI (Graphical User Interface) --------------------------

from ipywidgets import interact, interactive, fixed
import ipywidgets as wdgts 

def func(x):
    return x

interact(func, x = 10)



