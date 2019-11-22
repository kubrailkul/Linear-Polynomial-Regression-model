



from numpy import *
from matplotlib.pyplot import *
import pandas as pd
from scipy import linalg

data = pd.read_csv("C:/Users/pc/Desktop/kurs/HW1_DATA.csv")

x=data.x
y=data.y


plot(x,y,'.')
show()

xsum=0
ysum=0
yx=y.dot(x)

for i in range (0,1000): 
    xsum=xsum+x[i]
    ysum=ysum+y[i]
    
def mean(num):          
	return num/1000

den=mean(x.dot(x))-mean(xsum)**2 

a= mean(yx)-mean(xsum)*mean(ysum) 
a=a/den

b=mean(ysum)*mean(x.dot(x))-mean(yx)*mean(xsum) 
b=b/den

yhat=a*x+b    

plot(x,y,'.')
plot(x,yhat,'.') #We don't need to write these part. I wrote in order to compare linear and polynomail regression.
show()   

sren=0
stot=0
for i in range (0,999):     
      sren=sren+(y[i]-yhat[i])**2
      stot=stot+(y[i]-mean(ysum))**2
      
rsquare=1-sren/stot



stot=0
for i in range (0,999):
    stot=stot+(y[i]-mean(ysum))**2
    
    
def ypoly(p,x):
    yhat=0
    l=len(p)-1
    for i in range (0,len(p)):
        yhat=yhat+p[i]*x**(l-i)        
    return yhat

def rsquare(yhat,y,stot):
    sren=0
    for i in range (0,999):
      sren=sren+(y[i]-yhat[i])**2
    return 1-sren/stot



ones=np.ones(1000)

xhat= np.c_[x,ones]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)



xhat= np.c_[x**2,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)



xhat= np.c_[x**3,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)

xhat= np.c_[x**4,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)

xhat= np.c_[x**5,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)

xhat= np.c_[x**6,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)

xhat= np.c_[x**7,xhat]
w = np.linalg.solve(np.transpose(xhat).dot(xhat),np.transpose(xhat).dot(y) )
print(w)


y2=ypoly(w,x)    
a2=rsquare(y2,y,stot)
plot(x,y,'.',color='b')
plot(x,y2,'.',color='g')

