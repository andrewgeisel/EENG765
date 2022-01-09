import monopoly as monop
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Create a 1d numpy vector with a 1 in the first location
# and 0's everywhere else.  Call it "my_vec"

### Your code goes here
my_vec = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

### How does it look?
# Test my_vec
monop.plot_pdf(my_vec,color_bar=True)
plt.show()

### Your code goes here.  Read in 'M' and multiply it times my_vec
npzfile = np.load('TestMatAndVec.npz') # this is probably wrong
lst = npzfile.files
M = npzfile[str(list(lst)[0])]
v1 = npzfile[str(list(lst)[1])]
# Multiply M and my_vec together, then plot the result
times_one_vec = np.matmul(M,my_vec)
###Uncomment or copy this line below
monop.plot_pdf(times_one_vec,color_bar=True)
plt.show()

#Multiply M times your created vector 10 times, then plot the result
#### Your code goes here ####
times_ten_vec = np.matmul(np.linalg.matrix_power(M,10), my_vec)
monop.plot_pdf(times_ten_vec,color_bar=True)
plt.show()

### Now multiply M times the vector (twice!) from the file 
M_and_v = np.matmul(np.linalg.matrix_power(M,2), v1)
### You can test your output vector here
monop.plot_pdf(M_and_v,color_bar=True)
plt.show()

C = la.expm(np.zeros((3,3)))
print(C)
#### Store this matrix as 'C' and the vector just created to a file 'MyData.npz'
np.savez('MyData.npz', C=C, v=times_ten_vec)