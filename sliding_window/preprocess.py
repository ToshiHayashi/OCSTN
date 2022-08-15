import pandas as pd
import numpy as np
import pickle
def makefile(what,filename):
    with open(filename,"wb") as f3:
        pickle.dump(what,f3)

def readfile(filename):
    with open(filename,"rb") as f4:
        ans=pickle.load(f4)
    return ans


d=660
start=[0,90,150,210,270,360,450,510,570,630,690]
end=[60,120,180,240,300,420,480,540,600,660,720]
output=np.zeros([0,d])
for i in range(20):
    print(i)
    for ind in range(11):
        try:
            data=np.array(pd.read_csv("c"+str(i+1)+".csv",header=None)[0])[start[ind]*1000:end[ind]*1000]
            data=np.array([data[n:n+d] for n in range(len(data)-d)])
            output=np.concatenate([output,data],axis=0)
        except:
            pass
makefile(output,"normal_breath660.pkl")
