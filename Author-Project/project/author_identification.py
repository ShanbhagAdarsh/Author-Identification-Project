#============================Library==============================================
import cv2
import numpy as np
import easyocr
from PIL import Image
import csv
import time
import matplotlib.pyplot as plt
#===================================Functions======================================
def listToString(s): 
    str1 = "" 
    for ele in s: 
        str1 += ele  
    return str1 

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0 
               for i, j in zip(str1, str2)) / float(len(str1))
#==================================Execution Time====================================
start = time.time()
a = 0
for i in range(1000):
    a += (i**100)
#======================================Main code======================================
reader = easyocr.Reader(['en'],gpu = False) # load once only in memory
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
image_file_name='Dataset/a.png' 
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

image = cv2.imread(image_file_name)

# sharp the edges or image.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
r_easy_ocr=reader.readtext(thresh,detail=0)

extracted_text = listToString(r_easy_ocr)
print("Extracted----"+r_easy_ocr[1])
#===============================checking through CSV file==============================
author_prediction = [] 
execution_time = []
check = 0
with open("Testing_data_csv/Test_data.csv") as File:
    reader = csv.DictReader(File)
    for row in reader:
        print("CSV File----"+row['ID'])
        if row['ID'] == r_easy_ocr[1]:
            check = 1
            print("ID :"+r_easy_ocr[1])
            print("Author : "+row['NAME'])
            res = similar(row['TEXT'],extracted_text)
            res = float(res)*1000
            print ("Accuracy : " + str(res))
            end = time.time()
            print("Execution Time :", end-start)
            time = int(end-start)
            #===========================Graph Plotting====================================
            author_prediction.append(res)
            execution_time.append(time)
            X_axis = np.arange(len(execution_time))

            plt.bar(X_axis-0.2, author_prediction, 0.1, label = 'Author Prediction')
            plt.bar(X_axis, execution_time, 0.1, label = 'Execution Time')

            plt.xlabel("Author Name : "+row['NAME'])
            plt.ylabel("Accuracy")
            plt.title("Author Recognition ")
            plt.legend()
            plt.savefig("Graphs/"+row['NAME']+".png")
            break
    if check == 0:
        print("Not Found........")