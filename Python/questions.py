
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
