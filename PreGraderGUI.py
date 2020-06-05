# -*- coding: utf-8 -*-
"""
Created on Sun May 06 21:56:15 2020

@author: EsmaKrasniqi
"""


# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter.ttk import *
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score as score
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import MinMaxScaler


root=tk.Tk()
root.title("Grade Prediction System")
root.geometry('600x250')
#root.configure(bg='white')
root.resizable(width=False, height=False)
button = tk.Button(root,text="Predict Nota Mesatare", width=25,bg='#4B8BBE') # button to call the 'values' command above 

tk.Label(root, text="Gender:").grid(row=0)
tk.Label(root, text="Race/Ethnicity").grid(row=1)
tk.Label(root, text="parental level of education").grid(row=2)
tk.Label(root, text="lunch").grid(row=3)
tk.Label(root, text="test preparation course").grid(row=4)
tk.Label(root, text="Grade_math").grid(row=5)
tk.Label(root, text="Grade_reading").grid(row=6)
tk.Label(root, text="Grade_writing").grid(row=7)
tk.Label(root, text="Nota mesatare").grid(row=9)

tk.Label(root, text='0-Female, 1- Male').grid(row=0, column=2)
tk.Label(root, text='1-group A, 2-group B, 3-group C,\n 4-group D, 5-group E').grid(row=1, column=2)
tk.Label(root, text='0-some high school, 1-high school, 2-associates D\n,3-some college, 4-bachelors D, 5-masters D').grid(row=2, column=2)
tk.Label(root, text='0-free/reduced, 1- standard').grid(row=3, column=2)
tk.Label(root, text='0-completed, 1- none').grid(row=4, column=2)


e1=Combobox(root)
e1["values"]=("0","1")
e2=Combobox(root)
e2["values"]=("1","2","3","4","5")
e3=Combobox(root)
e3["values"]=('0','1','2','3','4','5')
e4=Combobox(root)
e4["values"]=("1","0")
e5=Combobox(root)
e5["values"]=("1","0")
e6=Combobox(root)
e6["values"]=("10","9","8","7","6","5")
e7=Combobox(root)
e7["values"]=("10","9","8","7","6","5")
e8=Combobox(root)
e8["values"]=("10","9","8","7","6","5")

e1.grid(row=0,column=1)
e2.grid(row=1,column=1)
e3.grid(row=2,column=1)
e4.grid(row=3,column=1)
e5.grid(row=4,column=1)
e6.grid(row=5,column=1)
e7.grid(row=6,column=1)
e8.grid(row=7,column=1)
button.grid(row=10,column=1)

students=pd.read_csv("output\students2.csv",encoding="cp1252")


students1 = pd.DataFrame(students,columns=['gender','race ethnicity','parental level of education','lunch','test preparation course',
                                   'Grade_math','Grade_reading','Grade_writing']) 
x = students1.iloc[:,:7] 
y = students1.iloc[:,7]
model = RandomForestClassifier()
def getInput(event):
    global model
    global students1
    global x,y
    students1 = {'gender':int(e1.get()),'race ethnicity':int(e2.get()),'parental level of education':int(e3.get()),'lunch':int(e4.get()),'test preparation course':int(e5.get()),'Grade_math':int(e6.get()),'Grade_reading':int(e7.get()),'Grade_writing':int(e8.get())}
    print(students1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)

# futja e te dhenave trajnuese ne model
    model.fit(x_train, y_train)
# Parashikimi i rezultateve ne test set
    filename = 'output/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    model=pickle.load(open(filename,'rb'))
    prediction = model.predict(x_test)
    result = model.score(x_test, y_test)
    print(result)
    tk.Label(root,text=prediction.mean()).grid(row=9,column=1)

button.bind("<Button>",getInput)

  
root.mainloop()
