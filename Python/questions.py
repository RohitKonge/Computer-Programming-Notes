
import re
import numpy as np
import pandas as pd
import csv

# data = list(csv.reader(open("Python/Utilities/EcommercePurchases.csv", mode="r")))

# ecom = pd.DataFrame(data[1:], columns=data[0])
# print(ecom.head(4))


def func(x):
    try:
        return float(x)
    except:
        pass
# print(sal["BasePay"].apply(lambda x : func(x)).mean())

# print(sal["OvertimePay"].apply(lambda x : func(x)).max())


# a = sal["EmployeeName"].values
# B = (np.where(a == "JOSEPH DRISCOLL")[0][0])

# print(sal.loc[B]["JobTitle"])

# print(sal.loc[B]["TotalPayBenefits"])
# c = str(sal["TotalPayBenefits"].apply(lambda x: func(x)).max())
# d = sal["TotalPayBenefits"].values
# e = np.where(d == c)
# print(sal.loc[e]["EmployeeName"])


# c = str(sal["TotalPayBenefits"].apply(lambda x: func(x)).min())
# d = sal["TotalPayBenefits"].values
# e = np.where(d == c)
# print(sal.loc[e]["EmployeeName"])


# sal["BasePay"] = sal["BasePay"].apply(lambda x : func(x))
# print(sal.groupby("Year")["BasePay"].mean())

# print(sal["JobTitle"].value_counts().head(5))


data = list(csv.reader(open("Python/Utilities/EcommercePurchases.csv", mode="r")))

ecom = pd.DataFrame(data[1:], columns=data[0])
# print(ecom.info())

a = ecom["Purchase Price"].apply(lambda x: func(x))
# print(a.mean())
# print(a.max())
# print(a.min())
# print(len(ecom[ecom["Language"] == "en"]))
# print(len(ecom[ecom["Job"] == "Lawyer"]))
# print(ecom["AM or PM"].value_counts())
# print(ecom["Job"].value_counts().head(5))
# print(ecom[ecom["Lot"] == "90 WT"]["Purchase Price"])
# print(ecom[ecom["Credit Card"] == "4926535242672853"]["Email"].values[0])
# print(len(ecom[(ecom["CC Provider"] == "American Express") & (ecom["Purchase Price"].apply(lambda x: func(x)) > 95) ]))
# print(len(ecom[ecom["CC Exp Date"].apply(lambda x : x[3:5]) == "25"]))
# print(ecom["Email"].apply(lambda x: x.split("@")[1]).value_counts().head(5))




import os, re
files = os.listdir("Computer-Programming-Notes/Python/Utilities/extracted_content")
print(files)
for i in files:
    if i.find("txt") < 0:
        txt_files = (os.listdir(f"Computer-Programming-Notes/Python/Utilities/extracted_content/{i}"))
        for j in txt_files:
            f = open(f"Computer-Programming-Notes/Python/Utilities/extracted_content/{i}/{j}","r")
            searched = re.search(r"\d{3}-\d{3}-\d{4}", list(f)[0])
            if searched != None :
                print(searched)
                break
        

import re
f = open("Computer-Programming-Notes/Python/Utilities/extracted_content/One/HDOHZHFSTTK.txt","r")

# print(len(list(f)[0]))


print(re.search(r"\d{3}-\d{3}-\d{4}", list(f)[0]))