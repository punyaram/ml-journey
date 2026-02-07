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
def pred_prob(hours,w,b):
    sigmoid(w*hours+b)
def find_hours_for_95()