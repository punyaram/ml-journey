#1.matrix multiplication
def shape(l):
    return (len(l),len(l[0]))
def matrixmul(p,q):
    rows_p=len(p)
    cols_p=len(p[0])
    cols_q=len(q[0])
    result= [[0 for _ in range(rows_p)] for _ in range(cols_q)]
    for i in range(rows_p):
        for j in range(cols_q):
            for k in range(cols_p):
                result[i][j]+=p[i][k]*q[k][j]
    return result
print(matrixmul([[1,2],[3,4]],[[5,6],[7,8]]))
x=[[1,2],[3,4]]
y=[[1,2,3],[4,5,6]]
print(shape(y))
#2.simple linear regression
def train(x,y,w,b,lr,epochs):
    n=len(x)
    for _ in range(epochs):
        #predict
        y_predict= [xi*w+b for xi in x]
        #computing dw and db
        dw=(2/n)*sum(x[i]*(y_predict[i]-y[i]) for i in range(n))
        db=(2/n)*sum(y_predict[i]-y[i] for i in range(n))
        #update
        w=w-lr*dw
        b=b-lr*db
    return w,b
print(train([28,30,32,34],[117.3,126.4,132.1,142.3],2,4,0.0001,100000))
#3.logistic regression-sigmoid curve
import math
def train(x,y,w,b,lr,epochs):
    n=len(x)
    for i in range(epochs):
        y_predict=[1/(1+math.exp(-(w*xi+b)))for xi in x]
        dw=(1/n)*sum(x[i]*(y_predict[i]-y[i]) for i in range(n))
        db=(1/n)*sum(y_predict[i]-y[i] for i in range(n))
        w=w-lr*dw
        b=b-lr*db
    return w,b
x=[1,2,3,4,5,6]
y=[0,0,0,1,1,1]
w,b=train(x,y,0,0,0.01,10000)
# print(w,b)
def predict(x,w,b):
    return [1/(1+math.exp(-(w*xi+b)))for xi in x]
test_x=[5.5,2.5,1.5]
print(predict(test_x,w,b))
def dot(v,w):
    return sum(v[i]*w[i] for i in range(len(v)))
def train(y,x,w,b,lr,epochs):
    n=len(x)
    m=len(w)
    for i in range(epochs):
        y_pred=[dot(xi,w)+b for xi in x]
        dw=[0]*m
        db=0
        for i in range(n):
            for j in range(m):
                dw[j]+=x[i][j]*(y_pred[i]-y[i])
            db+=y_pred[i]-y[i]
        for j in range(m):
            w[j]=w[j]-lr*(2/n)*dw[j]
        b=b-lr*(2/n)*db
    return w,b
x=[[1.44,2.32],[2.34,3.44],[3.23,4.18]]
w=[1,2]
y=[5.12,8.21,11.43]
b=0
print(train(y,x,w,b,0.0000001,30000))
