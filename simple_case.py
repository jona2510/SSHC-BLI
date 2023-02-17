"""
This code belongs to Semi-Supervised Hierarchical Classifier Based on Local Information
	SSHC-BLI: https://github.com/jona2510/SSHC-BLI

The SSHC-BLI is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

from SSHC_BLI.HSSL import SSHCBLI as bli
from SSHC_BLI.hStructure import hierarchy
from SSHC_BLI.LCN import TopDown as td
from SSHC_BLI.hStructure import policiesLCN as pol
import SSHC_BLI.evaluationHC as ehc
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.ensemble import RandomForestClassifier as rfc
from scipy import stats
import numpy as np
from time import time
from sklearn.datasets import make_moons



np.random.seed(0)


st = np.zeros((9,9),dtype=int)
st[0,3] = 1
st[0,4] = 1
st[2,5] = 1
st[2,6] = 1
st[3,7] = 1
st[3,8] = 1


H = hierarchy(st)
H.initialize()


# create artificial datatsets
# two atributes

ni=8
#nodes 7-8
data = np.zeros((ni,3))
data[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 0 )[0]
y[ind] = 7
ind = np.where( y == 1 )[0]
y[ind] = 8
data[:,2] = y[:]


#nodes 1-4
dt = np.zeros((ni,3))
dt[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 1 )[0]
y[ind] = 4
ind = np.where( y == 0 )[0]
y[ind] = 1
dt[:,2] = y[:]
dt[:,0] = dt[:,0] + 3.5
data = np.concatenate([data,dt])


#nodes 5-6
dt = np.zeros((ni,3))
dt[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 0 )[0]
y[ind] = 5
ind = np.where( y == 1 )[0]
y[ind] = 6
dt[:,2] = y[:]
dt[:,0] = dt[:,0] + 2.5
dt[:,1] = dt[:,1] - 1.5
data = np.concatenate([data,dt])


# shufle data
np.random.shuffle(data)

# generate classes:
cl = np.zeros((len(data),9),dtype=int)
for i in range(len(data)):
	cl[i] = H.getSinglePaths()[int(data[i,2])]

#remove last column from data
data = np.delete(data,[2],axis=1)


# generate unlabelled data
#node 7-8
ni = 200
unlabelled, y = make_moons(n_samples=ni, noise=0.1)

#node 1,4
un, y = make_moons(n_samples=ni, noise=0.1)
un[:,0] = un[:,0] + 3.5

unlabelled = np.concatenate([unlabelled,un])

#node 5-6
un, y = make_moons(n_samples=ni, noise=0.1)
un[:,0] = un[:,0] + 2.5
un[:,1] = un[:,1] - 1.5

unlabelled = np.concatenate([unlabelled,un])


# generate Test
ni=100
#nodes 7-8
test = np.zeros((ni,3))
test[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 0 )[0]
y[ind] = 7
ind = np.where( y == 1 )[0]
y[ind] = 8
test[:,2] = y[:]


#nodes 1-4
dt = np.zeros((ni,3))
dt[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 1 )[0]
y[ind] = 4
ind = np.where( y == 0 )[0]
y[ind] = 1
dt[:,2] = y[:]
dt[:,0] = dt[:,0] + 3.5
test = np.concatenate([test,dt])


#nodes 5-6
dt = np.zeros((ni,3))
dt[:,:2], y = make_moons(n_samples=ni, noise=0.1)
ind = np.where( y == 0 )[0]
y[ind] = 5
ind = np.where( y == 1 )[0]
y[ind] = 6
dt[:,2] = y[:]
dt[:,0] = dt[:,0] + 2.5
dt[:,1] = dt[:,1] - 1.5
test = np.concatenate([test,dt])

# shufle data
#np.random.shuffle(data)

# generate classes:
clt = np.zeros((len(test),9),dtype=int)
for i in range(len(test)):
	clt[i] = H.getSinglePaths()[int(test[i,2])]

#remove last column from data
test = np.delete(test,[2],axis=1)



sspol = pol(H,balanced=False,policy="lessInclusive")



vrs = ["v1","v2","v3"]
evm = ["EM","hR","hP","hF","MCC"]
res = [[] for x in range(len(evm))]

for variant in vrs:
	print("SSHC-BLI "+variant)
	bc = rfc(random_state=0)
	hc = bli(H,Hclassifier=td(H,baseClassifier=bc,policy=sspol),variant=variant,maxIterations=-1,threshold=0.5,k=3,t2considerNode=0.5,inck=5)

	hc.fit(data,cl,unlabelled)
	p = hc.predict(test)

	res[0].append(ehc.exactMatch(clt,p))
	res[1].append(ehc.hRecall(clt,p))
	res[2].append(ehc.hPrecision(clt,p))
	res[3].append(ehc.hFmeasure(clt,p))
	mlmcc = ehc.mlc_MCC(clt,p)
	res[4].append(np.average(mlmcc) )

	np.random.seed(0)


# supervised
print("supervised (TD)")
tdc = td(H,baseClassifier=rfc(random_state=0),policy=sspol,TL="from_scratch")
tdc.fit(data,cl)

p = tdc.predict(test)

print("*************************")
print("Results: ")
print("\t","TD\t","\t".join(vrs))
print("EM\t",ehc.exactMatch(clt,p),"\t".join([str(x) for x in res[0]]) )
print("hR\t",ehc.hRecall(clt,p),"\t".join([str(x) for x in res[1]]))
print("hP\t",ehc.hPrecision(clt,p), "\t".join([str(x) for x in res[2]]))
print("hF\t",ehc.hFmeasure(clt,p), "\t".join([str(x) for x in res[3]]))
mlmcc = ehc.mlc_MCC(clt,p)
print("MCC\t",np.average(mlmcc), "\t".join([str(x) for x in res[4]]) )



