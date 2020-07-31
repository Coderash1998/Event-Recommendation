import csv
import EventRecommender as er
from openpyxl import Workbook
book=Workbook()
sheet=book.active
sheet.append(["Event","Names"])
name=input("Enter filename with extension (.csv): ")
with open(name, mode ='r')as file:  
    csvFile = csv.reader(file) 
    for lines in csvFile: 
        d,e=er.inputline(lines[0])
        sheet.append([d,e])
book.save("Output.xlsx")
print("Recommendations generated successfully!!")