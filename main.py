import numpy as np
import cv2
list1 = [23,666,688,99,77,53,43,56,55]
print(list1[2:])

np1 = np.array([[1,24,4],[23,45,67]])
print(np1)
#print(np1[0])
#print(np1[0:])
#print(np1[0:1])
print(np1[:,2:])
#print(np1[1:2])

#print(np1[:0])

#for i in np1[0]:
    #print(i)

print("ram")

sample_img_array = [[[0,0,0],[0,10,0],[1,1,1]],
         [[0,0,0],[0,20,0],[2,23,2]]]

image_array= np.array(sample_img_array)

#print(image_array)
print(image_array[:2,1:,0:2])
#print(image_array[:1,1:2,1:2])
#print(image_array[:1,1:2,1:2])

#image_process

img1 = cv2.imread('files/image2.png',1)
cv2.imshow('emage',img1)


print(img1.shape)
#print(img1)

img2 = img1[:30,:30,:]
blue_image =  img1[:30,:30,0:1]
green_image = img1[:30,:30,1:2]
red_image =   img1[:30,:30,2:]

cv2.imshow('cropped',img2)
cv2.imshow('blue_image',blue_image)
#cv2.waitKey(5000)


x = open('files/ekdm new.txt','w')
x2 = open('files/blue.txt','w')
x3 = open('files/green.txt','w')
x4 = open('files/red.txt','w')

x.write(str(img2))
x2.write(str(blue_image))
x3.write(str(green_image))
x4.write(str(red_image))

x.close()
x2.close()
x3.close()
x4.close()

image_array = np.array([[2],[3],[4]])
#filter = np.array([[1],[0],[-1]])
filter2 = np.array([[[1],[2],[3]]])
filter = filter2

print(image_array.shape)
print(filter.shape)
print(filter.ndim)
product = np.multiply(image_array,filter)
print('##'*10)
print(product)
print(product.shape)

array1 = np.array([[1,2,3]])
print(array1.shape)
print(array1)
array2 = np.array([[2],[3],[4]])
print(array2)
print(array2.shape)
array4 = np.array([[[4],[3]],[[2],[3]]])
array3 = array1 * array4
print(array3)
print(array3.shape)

list1 = [10,20]
for num,value in enumerate(list1,start=0):
    print(num)
sum1 = np.sum(array4)
print(sum1)
array5 = np.array([[[[4]],[[5]]]])
array6 = np.array(array5,ndmin=2)
print(array5)
print(array6)

array7 = np.array([[[0],[1]]])
print(array7)
array7 = array7[0]
print(array7)
#print(arra)
f = [[[4,5,6],[5,6,7]]]
array8 = np.array(f)
print(array8.ndim,array8)
array9 = array8[0]
print(array9.ndim,array9)
print(array9.tolist())

x = array8/10
print(x)
y = x+ 3
print(y)

#checeking insert 

array10_list  = [[[12],[13]],[[15],[16]]]
array10 = np.array(array10_list)
print(type(array10))
array10 = array10.tolist()

array10.insert(0,[[134],[122]]) 
array10.append([[99],[97]])
print(array10) 

print(type(array10))
      

#checking maths on numpy arrays

array11= np.array([[4],[3],[2]])
array12 = np.exp(array11)
print(array12)
x1 = 1/(1+np.exp(array11))
print(x1)
array13 = array11 ** 2
print(array13)

class my_process:
    def check(self):
        self.i1 = 0
        xlist = [1,2,3,45,5]
        for i in xlist:
            print(self.i1,i)
            self.i1 = i
            print('    m  ',self.i1,i)


c1 = my_process()
c1.check()

#print(np.add(array11)) error add takes two arguments

#comparing whole array with specifc number and assigning a number

array14_d = [[[34,667,77],[32,67,8]],[[4,7,7],[67,97,9]]]

array14 = np.array(array14_d)
print(array14)
      
#array14[array14 > 36] = 1
b = np.where(array14 > 30 ,1,0)
print(b)
     
        





