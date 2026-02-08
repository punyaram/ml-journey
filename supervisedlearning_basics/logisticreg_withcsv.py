##MY INITIAL APPROACH-->
# import math
# import csv
# def load_data(file):
#     x=[]
#     y=[]
#     with open(file,'r') as f:
#         reader=csv.reader(f)
#         next(reader)
#         for row in reader:
#             x.append(float(row[0]))
#             y.append(int(row[1]))
#     return x,y
# def sigmoid(z):
#     return 1/(1+math.exp(-z))
# def train(x,y,w,b,lr,epochs):
#     n=len(x)
#     for _ in range(epochs):
#         y_pred=[sigmoid(w*xi+b) for xi in x]
#         dw=(1/n)*sum(x[i]*(y_pred[i]-y[i]) for i in range(n))
#         db=(1/n)*sum(y_pred[i]-y[i] for i in range(n))
#         w=w-lr*dw
#         b=b-lr*db
#     return w,b,y_pred
# def pred_prob(hours,w,b):
#     return sigmoid(w*hours+b)
# logsum=0
# def find_hours_for_95(y_pred):
#     for yi in y_pred:
#         logsum+=math.log(yi,2)
#         if(logsum>=0.95):
#             return y_pred
        
# x,y,=load_data("E:/academics/coursework/machine learning/study_hours.csv")
# w,b,y_pred=train(x,y,0,0,0.01,3000)
# print(find_hours_for_95(y_pred))
#THE CORRECT METHOD
import math
import csv
def load_data(file):
    x=[]
    y=[]
    with open(file,'r') as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(int(row[1]))
    return x,y
def sigmoid(z):
    return 1/(1+math.exp(-z))
def train(x,y,w,b,lr,epochs):
    n=len(x)
    for _ in range(epochs):
        y_pred=[sigmoid(w*xi+b) for xi in x]
        dw=(1/n)*sum(x[i]*(y_pred[i]-y[i]) for i in range(n))
        db=(1/n)*sum(y_pred[i]-y[i] for i in range(n))
        w=w-lr*dw
        b=b-lr*db
    return w,b
def find_hours_for_95(w,b):
    hours=0
    while hours<=24:
        prob=sigmoid(w*hours+b)
        if prob>=.95:
            return hours
        hours +=0.1
    return None
x,y,=load_data("E:/academics/coursework/machinelearning/study_hours.csv")
w,b=train(x,y,0,0,0.01,11100)
print(find_hours_for_95(w,b))