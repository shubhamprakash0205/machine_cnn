"""
x = 3
with open(f'files/test_result/{x}.txt','w') as xfile:
    xfile.write(f'Rama')
    xfile.close
"""

import cv2
import numpy as np
import ast
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

x= cv2.imread('files/images/test/1.jpg')
print(x.shape)
t= x.squeeze()
print(t.shape)
x = x[:,:,:1]
print(x.shape)
f = x.squeeze()
print(f.shape)
#cv2.imwrite('files/test_result/nothing2.jpg',x)

array1 = [[3,5],[5,6],[46,65]]
array1 = np.array(array1)

array2 = [[3],[4],[5]]
array2 = np.array(array2)

array3 = array1 * array2
print(array3)
print(array3[:,:1])
array4 = array3.tolist()
print(array3)
print(array4)
print(np.sum(array1))



def myconvolution(ximage,ximage_channels,filter):
	filter_length=len(filter)
	list_y = []
	for i_y in range(len(ximage)-filter_length+1):
		list_x = []
		for i_x in range(len(ximage[0])-filter_length+1):
			processing_ximage = ximage[i_y:i_y+filter_length,i_x:i_x+filter_length,:] #shape(3,3,no_channels)
			ss = []
			for f in range(filter_length):
				s = processing_ximage[f] * filter[f]
				ss.append(s.tolist())
			ss_array = np.array(ss) #shape(3,3,no_channels)
			
			rcc = 0
			fcc = 0
			for rc in ss_array:
				rcc = rc + rcc
			#rcc_shape(3,no_channels)
			
			for fc in rcc:
				fcc = fc + fcc
			#fcc_shape(1)
			
			list_x.append(fcc)
						
		list_y.append(list_x)
	return np.array(list_y)#output_shape_is(height,width,no_channel)


			
#for pooling
def mypooling(img,num_channel):
    img = img.tolist() if str(type(img)) == "<class 'numpy.ndarray'>" else img
    element_list=[]
    for i in range(num_channel):
         element_list.append(0)
        
    for i,element in enumerate(img,start = 1):
        element.insert(0,element_list)
        element.append(element_list)
           
            
    listxx = []
    for i in range(len(img[0])):
        listxx.append(element_list)
                    
    img.insert(0,listxx)
    img.append(listxx)
    
    return np.array(img)

	
#f11 = cv2.imread(f'files/images/test/2.jpg')[:,:,:1]
#print(f11.shape)
#f1 = mypooling(f11,1)
#fm = f1
#print(f1.shape)
#f22 = np.array([[[1],[1],[1],[1],[1]],[[0],[0],[0],[0],[0]],[[-1],[-1],[-1],[-1],[-1]],[[-1],[-1],[-1],[-1],[-1]],[[-1],[-1],[-1],[-1],[-1]]])
#f2 = np.array([[[1],[0],[-1]],[[1],[0],[-1]],[[1],[0],[-1]]])
#f3 = myconvolution(f1,1,f2)
#print(f3.shape)

"""
with open('files/f1.txt','w') as f:
	f.write(str(f1[:2,:2,:]))
	f.close()
with open('files/f3.txt','w') as f2:
	f2.write(str(f3[:2,:2,:]))
	f2.close()
"""

#cv2.imshow('dd',f11)
#cv2.imshow('chehe',f3/(3))

#cv2.waitKey(0)


print('fffffff*10')
array1 = np.random.random((2,2,))

print(array1)
array2 = np.log(array1)
print(array2)
for i in range(500):
	print(i,end='\r')


x1 = np.array([0.00,1,5,0])
y1 = np.where(x1==0.0,0.0001,x1)
y1 = np.where(y1 == 5,55,y1)
print(x1)
print(y1)
t = 8%2
print(t)

"""
import os

f = os.listdir(f'files/ground_truth/test')
ft = 'files/ground_truth/test'
mylist = []
for x in f:

	y = x[:len(x)-4]
	print(x," ",y)
	mylist.append(int(y))
	mylist.sort()
	
	#print(type(x))

for i in range(1,len(mylist)+1):
	fg = f'{mylist[i-1]}.mat'
	fgg = f'{i}.mat'
	os.rename(os.path.join(ft,fg),os.path.join(ft,fgg))
"""

def myfull_convolution(dz,w):
	w = w[::-1,::-1,:]
	l = len(dz) + len(w) - 1
	h = len(dz[0]) + len(w) - 1
	c = len(dz[0][0])
	a = np.zeros(shape=(l,h,c))
	print('operation started:::')
	
	for i_y in range(len(dz)):
		for i_x in range(len(dz[0])):
			print('i_x:',i_x,a,'ended')
			s = dz[i_y][i_x] * w
			print(s,'ended')
			ax = a[i_y:i_y+3,i_x:i_x+3] + s
			a[i_y:i_y+3,i_x:i_x+3] = ax
	return a


array1 = [[[1,2,3],[2,4,6],[3,4,5],[4,5,6]],[[5,6,7],[6,7,8],[7,8,9],[8,9,10]],[[9,10,11],[10,11,12],[11,12,13],[12,13,14]]]
array2 = [[[9],[8],[7]],[[6],[5],[4]],[[3],[2],[1]]]
array1 = np.array(array1)
array2 = np.array(array2)
print(array2,'with shape ',array2.shape)
print(array2.T,'with shape',array2.T.shape)
array3 = (array2.T).T
print(array3,'with shape is ',(array2.T).T.shape)
#array2 = array2[::-1,::-1,:]
#print(array2)

c = myfull_convolution(array1,array2)
print(c,'c')


def anti_pooling(a):
	#aa = np.delete(a,[0,len(a)-1],axis=0)
	#aaa = np.delete(aa,[0,len(aa[0])-1],axis=1)
	aa = a[1:len(a)-1,1:len(a[0])-1]
	return aa
"""
f = cv2.imread('files/images/test/1.jpg')
cv2.imshow(f'{f.shape}',f)
ff = mypooling(f,3)
cv2.imshow(f'pooled{ff.shape}',ff/2)
fff= anti_pooling(ff)
cv2.imshow(f'antiPOOled{fff.shape}',fff/2)
cv2.waitKey(0)
"""
array4 = np.array([[1,2,3,4,5,6,7],[1,2,34,5,6,7,8],[1,23,4,55,67,8,8]])
array4 = np.sum(array4,axis =0)
print(array4)
import os
"""
#np.savez('files/test_result/mycheckpoint.npz',arrayx = 'ravan' ,arrayz= array1)
if os.path.exists('files/test_result/mycheckpoint.npz'):
	checkpoint = np.load('files/test_result/mycheckpoint.npz')
	print(type(checkpoint['arrayx']),'heheehe')
else:
	print(False)
"""	
s = np.array([34])
z = s % 3
print(z,type(z))
if z == np.array([1]):
	print('worl')
	
	
array5 = np.random.randint(low=0,high= 10,size = (2,3,3,1))	
array6 = np.random.randint(low=0,high= 10,size = (2,3,3,1))	

print(array5)
#array5 = np.sum(array5,axis =3)
print(array5)
array7 = [array5,array6]
array7 = np.array(array7)
print(array7)
"""
checkpoint = np.load('files/test_result/mycheckpoint.npz')
print(array1.shape,checkpoint['arrayz'].shape)
array3 = checkpoint['arrayz']
print(array3.shape)

array1 = checkpoint['arrayx']
if array1 == 'ravan':
	print('www')
if 1:
	print(array1,type(array1))
	x = array1.tolist()
	print(x)
"""
#print(np.show_config())
"""
import jax
import jax.numpy as jnp

print(jax.devices())	
"""

b = np.random.randint(low = 0,high=10,size=(3,3,3))
print(b)
b = np.sum(b,axis = 2)
c = np.sum(b,axis =0 ,keepdims= True)
print(b)
np.set_printoptions(threshold = np.inf)

"""
dc = np.load('files/test_result/7pixels/checkpoint -17.npz')

for k in dc.keys():
	
	print(k,type(dc[k]))
	
print(dc['pooled_a1'].shape)
print(dc['pixel1s_w'].shape)
print(dc['pixel2s_w'].shape)
f = dc['forward_state']
print(f,type(f),f.shape)
ff = dc['processing_stage']
print(ff,type(ff),ff.shape,ff.ndim)
"""

#f = cv2.imread('files/images/test/1.jpg')
#r = mypooling(f,3)
#ff = anti_pooling(r).astype(np.float32)

#cv2.imshow(str(f.shape),f/2)
#cv2.imshow(str(r.shape),r/2)
#cv2.imshow(str(ff.shape),ff)
#cv2.waitKey(0)

x = np.load('files/test_result/mycheckpoint.npz')
w = x['w']
b = x['b']

print(w)
print(b)


array1 = [5,6,7,8,9,55]
array1 = np.array(array1)
array2 = np.where(array1 <= 6,2*array1,array1)
print(array2)



from scipy.io import loadmat

da = loadmat('files/ground_truth/test/1.mat')
print(type(da))
print(da.keys())

x = da['groundTruth']
print(type(x))
print(x.ndim,x.shape) #2,(1,5)
print(len(x)) #1
#[[a,b,c,d,e]]

print('gt_list x[0]')
gt_list = x[0]
print(type(gt_list))
print(gt_list.ndim,gt_list.shape) #1,(5,)
print(len(gt_list)) #5
#[a,b,c,d,e]

print('gt_list elements now')
print(type(gt_list[0]))
print(gt_list[0].ndim,gt_list[0].shape)
print(len(gt_list[0]))
#[[alpha]]
#print(gt_list[0])
x = open('files/test.txt','w')
x.write(str(gt_list[4]))
print("gt[0]['boundry']Now boundries")
el = gt_list[1]['Boundaries']
print(type(el))
print(el.ndim,el.shape)
print(len(el))
print(np.max(el[0][0]))

#cv2.imshow('grd',el[0][0]*255)
cv2.waitKey(0)
ccd = []
for i in range(5):
	ccd.append(gt_list[i]['Boundaries'][0][0])
	#print(ccd.shape,'ffdd')
#cc = np.array(ccd)
#print(cc.shape,'ewer')
cc = np.sum(ccd,axis=0,keepdims= True)/5
print(cc.shape,np.max(cc))
"""
for i,element in enumerate(gt_list,start=1):
    imgg = element['Boundaries'][0][0]
    with open ('files/Pground.txt','w') as f:
        f.write(str(imgg))    
    print('   groudn_truth    ',imgg.shape)
    plt.subplot(1,len(gt_list)+1, i)
    cv2.imshow('ddd',imgg)
    cv2.waitKey(0)
    plt.imshow(imgg,cmap='gray')
    plt.title(f'ground_truth {i}')
    plt.axis('off')

plt.subplot(1,7,7)
plt.imshow(arrayf,cmap='gray')  
plt.title(f'{len(arrayf)},{len(arrayf[0])}') 
plt.axis('off')
#plt.show()
"""
ds = np.array([[[4],[5],[6],[74]]])
#1,4,1
#print(ds.shape)
#np.show_config()

array12 = np.array([[2,3],[4,5],[6,7]])
array13 = np.array([[1,2,3,4],[5,6,7,8]])
print(array12.shape)
print(array13.shape)
#print(array12 * array13)
print()
print(array12 @ array13)

x= np.random.randint(0,10,(8,1))
y= np.random.randint(0,10,(3,))
z= np.random.randint(0,10,3)

print(np.random.random((1,3)))
print(x)
print(y)
print(z)

t = np.argmax(x)
print(t)

x2 = np.array([[3,4],[4,5],[4,1]])
print(np.argmax(x2),x2.shape,np.max(x2))

x3 = np.array([[1,2],[1,2],[1,3]])
print(x2/x3)


class test:
	def mai(self,arg):
		print('fff')
	def mai2(self,t,t2):
		print(t,t2)
	def mai3(self,t5):
		print(t,t5)
	
	def mai4(self):
		self.mai('ravan')
		self.mai2(2,3)
		self.mai3(6)
		
f = test()
f.mai4()


