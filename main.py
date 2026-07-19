import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from numba import njit


#for padding

def mypadding(img,num_channel):
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


def anti_padding(a):
	#aa = np.delete(a,[0,len(a)-1],axis=0)
	#aaa = np.delete(aa,[0,len(aa[0])-1],axis=1)
	aa = a[1:len(a)-1,1:len(a[0])-1]
	return aa


def myconvolution(ximage,filter):
	#print(ximage.shape,filter.shape)
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
	return np.array(list_y)#output_shape_is(height-2,width-2,no_channel)


#for backpropagation

def myfull_convolution(dz,w):
	#print(f'shapes under myfull_convolution: {dz.shape} and {w.shape}')
	w = w[::-1,::-1,:]
	l = len(dz) + len(w) - 1
	h = len(dz[0]) + len(w) - 1
	c = len(dz[0][0])
	a = np.zeros(shape=(l,h,c))
	
	for i_y in range(len(dz)):
		for i_x in range(len(dz[0])):
			s = dz[i_y][i_x] * w
			ax = a[i_y:i_y+3,i_x:i_x+3] + s
			a[i_y:i_y+3,i_x:i_x+3] = ax
	return a
	
			


#coorelation
def mycoorelation(dz,a1):
	matrix_sum = 0
	#print('conv_process started')
	#print('shapes are',dz.shape,a1.shape)
	for x,x_data in enumerate(dz,start=0):
		#print(f"x is {x}")
		for y,y_data in enumerate(x_data,start=0):
			#print(f"y is {y}")
            
			matrix_sum = matrix_sum + y_data * a1[x:x+3,y:y+3]
			#print('matrix sum' , matrix_sum)
	
	matrix_sum = np.sum(matrix_sum,axis =2,keepdims=True)
	return matrix_sum
				

class sevenpixels:
	def __init__(self):
		self.original_img_dict = []
		self.grd_dict = []
		self.padded_a0 = []
		self.countt = np.array([1])
		self.learn_rate = 0.00000001
		
		for i in range(1,2):
			self.original_img_dict.append(cv2.imread(f'files/seven_pixels_training/17.jpg'))
		for i in range(1,2):
			self.grd_dict.append(cv2.imread(f'files/seven_pixels_training/17_grd.jpg'))
			
	
	def pixel1(self,pixel1_w,pixel1_b):
		#print(len(self.original_img_dict))
		if str(type(self.padded_a0)) != "<class 'numpy.ndarray'>":
			
			for img_num,img in enumerate(self.original_img_dict):
				a0 = img[:,:,::-1]
				a0 = a0/255
				#pixel1_z1 = layer1_a0 * layer1_w + layer1_b
				self.padded_a0.append(mypadding(a0,3))
			self.padded_a0 = np.array(self.padded_a0).astype(np.float32)
		
		
		a0 = self.padded_a0[0].astype(np.float32)	
		pixel1_z1 = myconvolution(a0,pixel1_w) + pixel1_b
		#print(f'max_value of a0: {np.max(a0)} and minimum value {np.min(a0)}')
		#print(f'max of z1 :{np.max(pixel1_z1)}, and min of z1 :{np.min(pixel1_z1)}')
		#pixel1_a1 = 1/(1+np.exp(-pixel1_z1))
		#1/(1+ np.round(np.exp(-z1),4))
		pixel1_a1 = np.where(pixel1_z1 <= 0,0.0001*pixel1_z1,pixel1_z1) #kind of leakyRelU
		pixel1_a1 = np.where(pixel1_a1 >= 10,np.sqrt(90+pixel1_a1),pixel1_a1) #underroot (90+a1)
		
		#print(f'max of a1: {np.max(pixel1_a1)} and minimum of a1 {np.min(pixel1_a1)}')
		return pixel1_z1,pixel1_a1
	
	def layer1(self):
		self.layer1_pixels_output_z1 = [] #layer1 7a1's
		self.layer1_pixels_output_a1 = []
		self.padded_a1 = []
		
		for i in range(7):
			layer1_data = self.pixel1(self.pixel1s_w[i],self.pixel1s_b[i])
			self.layer1_pixels_output_z1.append(layer1_data[0])
			self.layer1_pixels_output_a1.append(layer1_data[1])
			self.padded_a1.append(mypadding(layer1_data[1],3))
		
		self.layer1_z1 = np.array(self.layer1_pixels_output_z1)
		self.layer1_a1 = np.array(self.layer1_pixels_output_a1)
		self.padded_a1 = np.array(self.padded_a1)		
		
		return self.layer1_a1	
			
	def pixel2(self,pixel2_w,pixel2_b):
		#pixel2_w should have 7 filters of 3 * 3 . also pixel2_b should have 7 elements
		a1 = self.padded_a1 #[(height*width*3),.....] #7neurons
		print('its layer2::::')
		#print(len(a1),type(a1))
		#print(len(a1[0]),type(a1[0]))
		#print('heheh',a1.shape)
		pixel2_z2 = myconvolution(a1[0],pixel2_w[0]) + myconvolution(a1[1],pixel2_w[1]) + myconvolution(a1[2],pixel2_w[2]) + myconvolution(a1[3],pixel2_w[3]) + myconvolution(a1[4],pixel2_w[4]) + myconvolution(a1[5],pixel2_w[5]) + myconvolution(a1[6],pixel2_w[6])                                                            
		pixel2_z2 = pixel2_z2 + pixel2_b[0] + pixel2_b[1] + pixel2_b[2] + pixel2_b[3] + pixel2_b[4] + pixel2_b[5] + pixel2_b[6]	
		
		pixel2_a2 = np.where(pixel2_z2 <= 0,0.0001*pixel2_z2,pixel2_z2) #kind of leakyRelU
		pixel2_a2 = np.where(pixel2_a2 >= 10,np.sqrt(90+pixel2_a2),pixel2_a2) #underroot (90+a2)

		return pixel2_z2,pixel2_a2 #shape(height-2,width-2,no_channel)
	
	def layer2(self):
		self.layer2_pixels_output_z2 = []
		self.layer2_pixels_output_a2 = []
		self.padded_a2 = []
		
		for i in range(7):
			layer2_data = self.pixel2(self.pixel2s_w[i],self.pixel2s_b[i])
			self.layer2_pixels_output_z2.append(layer2_data[0]) #each self.pixel2s_w[i] will have 7 elements or filter of 3*3
			self.layer2_pixels_output_a2.append(layer2_data[1])
			self.padded_a2.append(mypadding(layer2_data[1],3))
		
		self.layer2_z2 = np.array(self.layer2_pixels_output_z2)	
		self.layer2_a2 = np.array(self.layer2_pixels_output_a2)
		self.padded_a2 = np.array(self.padded_a2)
		print(self.layer2_a2.shape)
		return self.layer2_a2
	
	def pixel3(self,pixel3_w,pixel3_b):
		a2 = self.padded_a2 #7neurons 
		pixel3_z3 = myconvolution(a2[0],pixel3_w[0]) + myconvolution(a2[1],pixel3_w[1]) + myconvolution(a2[2],pixel3_w[2]) + myconvolution(a2[3],pixel3_w[3]) + myconvolution(a2[4],pixel3_w[4]) + myconvolution(a2[5],pixel3_w[5]) + myconvolution(a2[6],pixel3_w[6])                       
		pixel3_z3 = pixel3_z3 + pixel3_b[0]+pixel3_b[1]+pixel3_b[2]+pixel3_b[3]+pixel3_b[4]+pixel3_b[5]+pixel3_b[6]	
		
		pixel3_a3 = np.where(pixel3_z3 <= 0,0.0001*pixel3_z3,pixel3_z3) #kind of leakyRelU
		pixel3_a3 = np.where(pixel3_a3 >= 10,np.sqrt(90+pixel3_a3),pixel3_a3)
		return pixel3_z3,pixel3_a3 #shape(height-2,width-2,3)
	
	def layer3(self):
		self.layer3_pixels_output_z3 = []
		self.layer3_pixels_output_a3 = []
		self.padded_a3 = []
		
		for i in range(7):
			layer3_data = self.pixel3(self.pixel3s_w[i],self.pixel3s_b[i])
			self.layer3_pixels_output_z3.append(layer3_data[0])
			self.layer3_pixels_output_a3.append(layer3_data[1]) #each self.pixel3s_w[i] will have 7 elements or filter of 3*3
			self.padded_a3.append(mypadding(layer3_data[1],3))
		
		self.layer3_z3 = np.array(self.layer3_pixels_output_z3)
		self.layer3_a3 = np.array(self.layer3_pixels_output_a3)
		self.padded_a3 = np.array(self.padded_a3)
		
		return self.layer3_a3
	
	def pixel4(self,pixel4_w,pixel4_b):
		a3 = self.padded_a3
			
		pixel4_z4 = myconvolution(a3[0],pixel4_w[0]) + myconvolution(a3[1],pixel4_w[1]) + myconvolution(a3[2],pixel4_w[2]) + myconvolution(a3[3],pixel4_w[3]) + myconvolution(a3[4],pixel4_w[4]) + myconvolution(a3[5],pixel4_w[5]) + myconvolution(a3[6],pixel4_w[6])                       
		pixel4_z4 = pixel4_z4 + pixel4_b[0]+pixel4_b[1]+pixel4_b[2]+pixel4_b[3]+pixel4_b[4]+pixel4_b[5]+pixel4_b[6]	
		
		pixel4_a4 = np.where(pixel4_z4 <= 0,0.0001*pixel4_z4,pixel4_z4) #kind of leakyRelU
		pixel4_a4 = np.where(pixel4_a4 >= 10,np.sqrt(90+pixel4_a4),pixel4_a4)
		return pixel4_z4,pixel4_a4 #shape(height-2,width-2,3)
		
	def layer4(self):
		self.layer4_pixels_output_z4 = []
		self.layer4_pixels_output_a4 = []
		self.padded_a4 = []
		
		for i in range(7):
			layer4_data = self.pixel4(self.pixel4s_w[i],self.pixel4s_b[i])
			self.layer4_pixels_output_z4.append(layer4_data[0])
			self.layer4_pixels_output_a4.append(layer4_data[1]) #each self.pixel3s_w[i] will have 7 elements or filter of 3*3
			self.padded_a4.append(mypadding(layer4_data[1],3))
		
		self.layer4_z4 = np.array(self.layer4_pixels_output_z4)
		self.layer4_a4 = np.array(self.layer4_pixels_output_a4)
		self.padded_a4 = np.array(self.padded_a4)
		
		return self.layer4_a4		
	
	def pixel5(self,pixel5_w,pixel5_b):
		a4 = self.padded_a4
		pixel5_z5 = myconvolution(a4[0],pixel5_w[0]) + myconvolution(a4[1],pixel5_w[1]) + myconvolution(a4[2],pixel5_w[2]) + myconvolution(a4[3],pixel5_w[3]) + myconvolution(a4[4],pixel5_w[4]) + myconvolution(a4[5],pixel5_w[5]) + myconvolution(a4[6],pixel5_w[6])                       
		pixel5_z5 = pixel5_z5 + pixel5_b[0]+pixel5_b[1]+pixel5_b[2]+pixel5_b[3]+pixel5_b[4]+pixel5_b[5]+pixel5_b[6]	
		
		pixel5_a5 = np.where(pixel5_z5 <= 0,0.0001*pixel5_z5,pixel5_z5) #kind of leakyRelU
		pixel5_a5 = np.where(pixel5_a5 >= 10,np.sqrt(90+pixel5_a5),pixel5_a5)
		return pixel5_z5,pixel5_a5 #shape(height-2,width-2,3)
		
	def layer5(self):
		self.layer5_pixels_output_z5 = []
		self.layer5_pixels_output_a5 = []
		self.padded_a5 = []
		for i in range(7):
			layer5_data = self.pixel5(self.pixel5s_w[i],self.pixel5s_b[i])
			self.layer5_pixels_output_z5.append(layer5_data[0]) #each self.pixel3s_w[i] will have 7 elements or filter of 3*3
			self.layer5_pixels_output_a5.append(layer5_data[1])
			self.padded_a5.append(mypadding(layer5_data[1],3))
		
		self.layer5_z5 = np.array(self.layer5_pixels_output_z5)
		self.layer5_a5 = np.array(self.layer5_pixels_output_a5)
		self.padded_a5 = np.array(self.padded_a5)
		
		return self.layer5_a5
	
	def pixel6(self,pixel6_w,pixel6_b):
		a5 = self.padded_a5 
		pixel6_z6 = myconvolution(a5[0],pixel6_w[0]) + myconvolution(a5[1],pixel6_w[1]) + myconvolution(a5[2],pixel6_w[2]) + myconvolution(a5[3],pixel6_w[3]) + myconvolution(a5[4],pixel6_w[4]) + myconvolution(a5[5],pixel6_w[5]) + myconvolution(a5[6],pixel6_w[6])                       
		pixel6_z6 = pixel6_z6 + pixel6_b[0]+pixel6_b[1]+pixel6_b[2]+pixel6_b[3]+pixel6_b[4]+pixel6_b[5]+pixel6_b[6]	
		
		pixel6_a6 = np.where(pixel6_z6 <= 0,0.0001*pixel6_z6,pixel6_z6) #kind of leakyRelU
		pixel6_a6 = np.where(pixel6_a6 >= 10,np.sqrt(90+pixel6_a6),pixel6_a6)
		return pixel6_z6,pixel6_a6 #shape(height-2,width-2,3)
		
	def layer6(self):
		self.layer6_pixels_output_z6 = []
		self.layer6_pixels_output_a6 = []
		self.padded_a6 = []
		
		for i in range(7):
			layer6_data = self.pixel6(self.pixel6s_w[i],self.pixel6s_b[i])
			self.layer6_pixels_output_z6.append(layer6_data[0])
			self.layer6_pixels_output_a6.append(layer6_data[1]) #each self.pixel3s_w[i] will have 7 elements or filter of 3*3
			self.padded_a6.append(mypadding(layer6_data[1],3))
		
		self.layer6_z6 = np.array(self.layer6_pixels_output_z6)
		self.layer6_a6 = np.array(self.layer6_pixels_output_a6)
		self.padded_a6 = np.array(self.padded_a6)
		
		return self.layer6_a6

	def pixel7(self,pixel7_w,pixel7_b):
		a6 = self.padded_a6  
		pixel7_z7 = myconvolution(a6[0],pixel7_w[0]) + myconvolution(a6[1],pixel7_w[1]) + myconvolution(a6[2],pixel7_w[2]) + myconvolution(a6[3],pixel7_w[3]) + myconvolution(a6[4],pixel7_w[4]) + myconvolution(a6[5],pixel7_w[5]) + myconvolution(a6[6],pixel7_w[6])                       
		pixel7_z7 = pixel7_z7 + pixel7_b[0]+pixel7_b[1]+pixel7_b[2]+pixel7_b[3]+pixel7_b[4]+pixel7_b[5]+pixel7_b[6]	
		pixel7_a7 = 1/(1+np.exp(-pixel7_z7))
		return pixel7_a7 #shape(height-2,width-2,3)
		
	def layer7(self):
		self.layer7_pixels_output_a7 = []
		self.padded_a7 = []
		for i in range(1):
			layer7_data = self.pixel7(self.pixel7s_w[i],self.pixel7s_b[i])
			self.layer7_pixels_output_a7.append(layer7_data) #each self.pixel3s_w[i] will have 7 elements or filter of 3*3
			self.padded_a7.append(mypadding(layer7_data,3))
		self.layer7_a7 = np.array(self.layer7_pixels_output_a7)
		self.padded_a7 = np.array(self.padded_a7)
		
		return self.layer7_a7
		
	
	def forward_update(self,state):
		if state == 1:
			self.layer1()
			self.status_update(forward_layer=1,countt_update=None)
		if state == 2:
			self.layer2()
			self.status_update(forward_layer=2,countt_update=None)
		if state == 3:
			self.layer3()
			self.status_update(forward_layer=3,countt_update=None)
		if state == 4: 
			self.layer4()
			self.status_update(forward_layer=4,countt_update=None)
		if state == 5:
			self.layer5()
			self.status_update(forward_layer=5,countt_update=None)
		if state == 6:
			self.layer6()
			self.status_update(forward_layer=6,countt_update=None)
		if state == 7:
			self.layer7()
			self.processing_stage = 'BACKWARD_C'
			self.forward_state = 1
			self.status_update(forward_layer=None,countt_update=None)
		
		
			
	def backward_update(self):
		#grd_output = self.grd_dict[0]
		#da7 = 
		#print(self.layer7_a7[0].shape,self.grd_dict[0].shape)
		#dz7 = a7-y
		#downside normalised the grd_dict also between 0 and 1 
		self.dz7 = (anti_padding(self.padded_a7[0]) - self.grd_dict[0]/255).astype(np.float32)
		#print(self.dz7.shape)
		
		#dw7 = dz7 * a6
		self.dw7 = [] #list of 7 elements where each element is a 3*3 after coorelation #with 3channels
		for i in range(7):
			self.dw7.append(mycoorelation(self.dz7,self.padded_a6[i]))		
		self.dw7 = np.array(self.dw7)
		self.db7 = self.dz7
		
		
		#da6 = dz7 * w7
		self.da6=[] #list of 7 elements with each element is height * width 
		for i in range(7):
			self.da6.append(myfull_convolution(self.dz7,self.pixel7s_w[0][i]))
		self.da6 = np.array(self.da6)
		#dz6 = da6 * (a6(1-a6))
		#dz6 = da6 * 0.0001 ,z6<0
		#dz6 = da6 * 1 ,z6>=0
		#dz6 = da6 * 1/2sq.root(90+x)
		
		self.dz6 = [] #list of 7elements. each element is (height-2 * width -2 * num_channel)
		for i in range(7):
			f = np.where(self.layer6_z6[i]>10,1/(2*np.sqrt(90+self.layer6_z6[i])),1)
			f = np.where(self.layer6_z6[i]<0,0.0001,f)
			self.dz6.append(anti_padding(self.da6[i]) * f)
		self.dz6 = np.array(self.dz6)
		#dw6 = dz6 * a5
		self.dw6 = [] # list of 7elements . again each element has 7 dw of size 3*3
		for y in range(7):
			self.dw6.append([])
			for i in range(7):
				self.dw6[y].append(mycoorelation(self.dz6[y],self.padded_a5[i]))
		self.dw6 = np.array(self.dw6)
		self.db6 = self.dz6
		
		
		
		#da5 = dz6 * w6
		self.da5 = []
		for y in range(7):
			self.da5.append([])	
			for i in range(7):
				self.da5[y].append(myfull_convolution(self.dz6[y],self.pixel6s_w[y][i]))
		self.da5 = np.array(self.da5)
		self.da5 = np.sum(self.da5,axis = 0)	# 7elements corrosponding to the 7units of a5. each element is (height*width*num_channel)
		#dz5 = da5 * (a5(1-a5))
		#dz5 = da5 * -1,z6<0
		#dz5 = da5 * 1,z6>=0
		self.dz5 = [] #7 elements, each being (height-2 * width - 2)
		for i in range(7):
			f = np.where(self.layer5_z5[i]>10,1/(2*np.sqrt(90+self.layer5_z5[i])),1)
			f = np.where(self.layer5_z5[i]<0,0.0001,f)
			self.dz5.append(anti_padding(self.da5[i]) * f)
		
		self.dz5 = np.array(self.dz5)
		#dw5 = dz5 * a4
		self.dw5 = [] # 7 elements , again each element has 7dws . each of size 3 * 3
		for y in range(7):
			self.dw5.append([])
			for i in range(7):
				self.dw5[y].append(mycoorelation(self.dz5[y],self.padded_a4[i]))
		self.dw5 = np.array(self.dw5)
		self.db5 = self.dz5
		
		#da4 = dz5 * w5
		self.da4 = [] 
		for y in range(7):
			self.da4.append([])
			for i in range(7):
				self.da4[y].append(myfull_convolution(self.dz5[y],self.pixel5s_w[y][i]))
		self.da4 = np.array(self.da4)
		self.da4 = np.sum(self.da4,axis=0) #7elements corrosponding to the 7 units of a4, each element being (height*width)
		#dz4 = da4 * (a4(1-a4))
		
		
		self.dz4 = []
		for i in range(7):
			f = np.where(self.layer4_z4[i]>10,1/(2*np.sqrt(90+self.layer4_z4[i])),1)
			f = np.where(self.layer4_z4[i]<0,0.0001,f)
			self.dz4.append(anti_padding(self.da4[i]) * f)
		
		self.dz4 = np.array(self.dz4) # 7elements, each being (height-2 * width -2)
		#dw4 = dz4 * a3
		self.dw4 = []
		for y in range(7):
			self.dw4.append([])
			for i in range(7):
				self.dw4[y].append(mycoorelation(self.dz4[y],self.padded_a3[i]))
		self.dw4 = np.array(self.dw4) # 7 elements corrosponding to the 7filters (w4). again each element has 7dws(size 3 * 3)corrosponding to the 7slices of each filter.
		self.db4 = self.dz4
		
		
		#da3 = dz4 * w4
		self.da3 = []
		for y in range(7):
			self.da3.append([])
			for i in range(7):
				self.da3[y].append(myfull_convolution(self.dz4[y],self.pixel4s_w[y][i]))
		self.da3 = np.array(self.da3)
		self.da3 = np.sum(self.da3,axis = 0) #7elements corrosponding to the 7 units of a3
		#dz3 = da3 * (a3(1-a3))
		
		
		self.dz3 = []
		for i in range(7):
			f = np.where(self.layer3_z3[i]>10,1/(2*np.sqrt(90+self.layer3_z3[i])),1)
			f = np.where(self.layer3_z3[i]<0,0.0001,f)
			self.dz3.append(anti_padding(self.da3[i]) * f)
		
		self.dz3 = np.array(self.dz3) #7elements , each being (height-2,width-2)
		#dw3 = dz3 * a2
		self.dw3 = []
		for y in range(7):
			self.dw3.append([])
			for i in range(7):
				self.dw3[y].append(mycoorelation(self.dz3[y],self.padded_a2[i]))
		self.dw3 = np.array(self.dw3)
		self.db3 = self.dz3
		
		
		#da2 = dz3 * w3
		self.da2 = []
		for y in range(7):
			self.da2.append([])
			for i in range(7):
				self.da2[y].append(myfull_convolution(self.dz3[y],self.pixel3s_w[y][i]))
		self.da2 = np.array(self.da2)
		self.da2 = np.sum(self.da2,axis = 0)
		#dz2 = da2 * (a2(1-a2))
		self.dz2 = []
		for i in range(7):
			f = np.where(self.layer2_z2[i]>10,1/(2*np.sqrt(90+self.layer2_z2[i])),1)
			f = np.where(self.layer2_z2[i]<0,0.0001,f)
			self.dz2.append(anti_padding(self.da2[i]) * f)
		
		self.dz2 = np.array(self.dz2)
		#dw2 = dz2 * a1
		self.dw2 = []
		for y in range(7):
			self.dw2.append([])
			for i in range(7):
				self.dw2[y].append(mycoorelation(self.dz2[y],self.padded_a1[i]))
		self.dw2 = np.array(self.dw2)
		self.db2 = self.dz2
		
		
		#da1 = dz2 * w2
		self.da1 = []
		for y in range(7):
			self.da1.append([])
			for i in range(7):
				self.da1[y].append(myfull_convolution(self.dz2[y],self.pixel2s_w[y][i]))
		self.da1 = np.array(self.da1)
		self.da1 = np.sum(self.da1,axis = 0) 
		#dz1 = da1 * (a1(1-a1))
		
		
		self.dz1 = []
		for i in range(7):
			f = np.where(self.layer1_z1[i]>10,1/(2*np.sqrt(90+self.layer1_z1[i])),1)
			f = np.where(self.layer1_z1[i]<0,0.0001,f)
			self.dz1.append(anti_padding(self.da1[i]) * f)
		
		self.dz1 = np.array(self.dz1)
		#dw1 = dz1 * a0
		self.dw1 = []
		for y in range(7):
			self.dw1.append([])
			for i in range(1):
				self.dw1[y].append(mycoorelation(self.dz1[y],self.padded_a0[i]))
		self.dw1 = np.array(self.dw1)
		self.db1 = self.dz1
		
		self.processing_stage = 'FORWARD_C'
	
	
	def status_update(self,forward_layer,countt_update):
		#print(f'self.countt: {self.countt} ; layer: {self.forward_state}_completed')
		self.show()
		self.countt = self.countt + 1 if countt_update != None else self.countt
		fx = 'files/test_result/7pixels/max_min_checkpoint.txt'
		ff = 'files/test_result/7pixels/xcheckpoint.npz'
		
		try:
			t = self.dw1 
		except:
			self.dw1 = [3]
			self.dw2 = [3]
			self.dw3 = [3]
			self.dw4 = [3]
			self.dw5 = [3] 
			self.dw6 = [3]
			self.dw7 = [3]
		else:
			pass
		
		
		with open(fx,'a') as d:
			d.write(f"""
			        
			        counttt: {self.countt} max_a0 = {(np.min(self.padded_a0),np.max(self.padded_a0))}
				max_a1 = {(np.min(self.padded_a1),np.max(self.padded_a1))}
				max_a2 = {(np.min(self.padded_a2),np.max(self.padded_a2))}
				max_a3 = {(np.min(self.padded_a3),np.max(self.padded_a3))}
				max_a4 = {(np.min(self.padded_a4),np.max(self.padded_a4))}
				max_a5 = {(np.min(self.padded_a5),np.max(self.padded_a5))}
				max_a6 = {(np.min(self.padded_a6),np.max(self.padded_a6))}
				max_a7 = {(np.min(self.padded_a7),np.max(self.padded_a7))}
				
				max_dw1 = {np.max(self.dw1)}
			     max_dw2 = {np.max(self.dw2)}
			     max_dw3 = {np.max(self.dw3)}
			     max_dw4 = {np.max(self.dw4)}
			     max_dw5 = {np.max(self.dw5)}
			     max_dw6 = {np.max(self.dw6)}
			     max_dw7 = {np.max(self.dw7)}
			     
			      """)
				
		np.savez_compressed(ff,pixel1s_w = self.pixel1s_w,
				pixel2s_w = self.pixel2s_w,
				pixel3s_w = self.pixel3s_w,
				pixel4s_w = self.pixel4s_w,
				pixel5s_w = self.pixel5s_w,
				pixel6s_w = self.pixel6s_w,
				pixel7s_w = self.pixel7s_w,
				pixel1s_b = self.pixel1s_b,
				pixel2s_b = self.pixel2s_b,
				pixel3s_b = self.pixel3s_b,
				pixel4s_b = self.pixel4s_b,
				pixel5s_b = self.pixel5s_b,
				pixel6s_b = self.pixel6s_b,
				pixel7s_b = self.pixel7s_b,
				countt = self.countt,
				forward_state = self.forward_state,
				processing_stage = self.processing_stage,
				padded_a0 = self.padded_a0,
				padded_a1 = self.padded_a1,
				padded_a2 = self.padded_a2,
				padded_a3 = self.padded_a3,
				padded_a4 = self.padded_a4,
				padded_a5 = self.padded_a5,
				padded_a6 = self.padded_a6,
				padded_a7 = self.padded_a7,
				layer1_z1 = self.layer1_z1,
				layer2_z2 = self.layer2_z2,
				layer3_z3 = self.layer3_z3,
				layer4_z4 = self.layer4_z4,
				layer5_z5 = self.layer5_z5,
				layer6_z6 = self.layer6_z6,	
			)
		
		if self.processing_stage == 'FORWARD_C' and forward_layer != None: 
			self.forward_state = forward_layer + 1
			self.forward_update(self.forward_state) 
		
	
	def loss(self):
		return 1000
		
		
	
	def real_update(self):
		while self.loss() > 100:
			if self.processing_stage == 'FORWARD_C':
				
				self.forward_update(self.forward_state)
			if self.processing_stage == 'BACKWARD_C':	
				self.backward_update()
			
			#0.00001 * 10000
			#clipping to control dws
			self.dw1 = np.clip(self.dw1,-10000,10000)
			self.dw2 = np.clip(self.dw2,-10000,10000)
			self.dw3 = np.clip(self.dw3,-10000,10000)
			self.dw4 = np.clip(self.dw4,-10000,10000)
			self.dw5 = np.clip(self.dw5,-10000,10000)
			self.dw6 = np.clip(self.dw6,-10000,10000)
			self.dw7 = np.clip(self.dw7,-10000,10000)
			self.db1 = np.clip(self.db1,-10000,10000)
			self.db2 = np.clip(self.db2,-10000,10000)
			self.db3 = np.clip(self.db3,-10000,10000)
			self.db4 = np.clip(self.db4,-10000,10000)
			self.db5 = np.clip(self.db5,-10000,10000)
			self.db6 = np.clip(self.db6,-10000,10000)
			self.db7 = np.clip(self.db7,-10000,10000)
			 						
			
			
			for y in range(7):
				#print(self.pixel1s_w[y].shape,self.dw1[y][0].shape,'backdrop one hehe')
				self.pixel1s_w[y] = self.pixel1s_w[y] - self.learn_rate * self.dw1[y][0]
			
			#print(self.pixel2s_w.shape,self.dw2.shape,'backdrop one hehe')
			#print(self.dw2[0][0])
			#print(self.db2[0][0])
			self.pixel2s_w = self.pixel2s_w - self.learn_rate * self.dw2
			self.pixel3s_w = self.pixel3s_w - self.learn_rate * self.dw3
			self.pixel4s_w = self.pixel4s_w - self.learn_rate * self.dw4
			self.pixel5s_w = self.pixel5s_w - self.learn_rate * self.dw5
			self.pixel6s_w = self.pixel6s_w - self.learn_rate * self.dw6
			self.pixel7s_w[0] = self.pixel7s_w[0] - self.learn_rate * self.dw7
			
			
			for i in range(7):
				
				self.pixel1s_b[i] = self.pixel1s_b[i] - self.learn_rate * np.sum(self.db1[i])
				self.pixel2s_b[i] = self.pixel2s_b[i] - self.learn_rate * np.sum(self.db2[i])
				self.pixel3s_b[i] = self.pixel3s_b[i] - self.learn_rate * np.sum(self.db3[i])
				self.pixel4s_b[i] = self.pixel4s_b[i] - self.learn_rate * np.sum(self.db4[i])
				self.pixel5s_b[i] = self.pixel5s_b[i] - self.learn_rate * np.sum(self.db5[i])
				self.pixel6s_b[i] = self.pixel6s_b[i] - self.learn_rate * np.sum(self.db6[i])
			self.pixel7s_b[0] = self.pixel7s_b[0] - self.learn_rate * np.sum(self.db7)
			
			self.status_update(forward_layer=None,countt_update='YUP')
			
			
		
	
		
	def show(self):
		try:
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a1[i])
				plt.title(f'layer1_a1_{i-1}')
			plt.show()
			
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a2[i])
				plt.title(f'layer2_a2_{i-1}')
			plt.show()
			
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a3[i])
				plt.title(f'layer3_a3_{i-1}')
			plt.show()
			
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a4[i])
				plt.title(f'layer4_a4_{i-1}')
			plt.show()
			
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a5[i])
				plt.title(f'layer5_a5_{i-1}')
			plt.show()
			
			for i in range(7):
				plt.subplot(1,7,i+1)
				plt.imshow(self.padded_a6[i])
				plt.title(f'layer6_a6_{i-1}')
			plt.show()
				
			plt.subplot(1,1,1)
			plt.imshow(self.padded_a7[0])
			plt.title('final _ A7')
			plt.show()
			
		
		except Exception as e:
			print(str(e))
		else:
			pass
			
			
				
	def initiate(self):
		if os.path.exists('files/test_result/7pixels/xcheckpoint.npz'):
			checkpoint = np.load('files/test_result/7pixels/xcheckpoint.npz') 
			self.pixel1s_w = checkpoint['pixel1s_w']
			self.pixel2s_w = checkpoint['pixel2s_w']
			self.pixel3s_w = checkpoint['pixel3s_w']
			self.pixel4s_w = checkpoint['pixel4s_w']
			self.pixel5s_w = checkpoint['pixel5s_w']
			self.pixel6s_w = checkpoint['pixel6s_w']
			self.pixel7s_w = checkpoint['pixel7s_w']
			self.pixel1s_b = checkpoint['pixel1s_b']
			self.pixel2s_b = checkpoint['pixel2s_b']
			self.pixel3s_b = checkpoint['pixel3s_b']
			self.pixel4s_b = checkpoint['pixel4s_b']
			self.pixel5s_b = checkpoint['pixel5s_b']
			self.pixel6s_b = checkpoint['pixel6s_b']
			self.pixel7s_b = checkpoint['pixel7s_b']
			self.countt = checkpoint['countt']
			self.processing_stage = checkpoint['processing_stage']
			self.forward_state = checkpoint['forward_state']
			self.padded_a0 = []
			self.padded_a1 = checkpoint['padded_a1']
			self.padded_a2 = checkpoint['padded_a2']
			self.padded_a3 = checkpoint['padded_a3']
			self.padded_a4 = checkpoint['padded_a4']
			self.padded_a5 = checkpoint['padded_a5']
			self.padded_a6 = checkpoint['padded_a6']
			self.padded_a7 = checkpoint['padded_a7']
			print(type(self.padded_a1),len(self.padded_a1),self.padded_a1.shape)
			self.layer1_z1 = checkpoint['layer1_z1']
			self.layer2_z2 = checkpoint['layer2_z2']
			self.layer3_z3 = checkpoint['layer3_z3']
			self.layer4_z4 = checkpoint['layer4_z4']
			self.layer5_z5 = checkpoint['layer5_z5']
			self.layer6_z6 = checkpoint['layer6_z6']
			
						
			#self.processing_stage = 'FORWARD_C'
			#self.forward_state = 6
			
			
			
		
		else:
			self.countt = 0
			self.processing_stage = 'FORWARD_C'
			self.forward_state = 1
			std = np.sqrt(2.0 /441 ) #for he initilization
			
			self.pixel1s_w = np.random.randn(7,3,3,1).astype(np.float32) * std
			
			"""
			e_w = np.array([[[5],[10],[-11]],[[15],[-6],[8]],[[1],[6],[-9]]])
			dss = []
			for i in range(7):
				dss.append(e_w)
			self.pixel1s_w = np.array(dss)
			"""
			
			self.pixel1s_b = np.zeros((7,1)).astype(np.float32)
			
			self.pixel2s_w = np.random.randn(7,7,3,3,1).astype(np.float32) * std
			self.pixel2s_b = np.zeros((7,7,1)).astype(np.float32)
			
			self.pixel3s_w = np.random.randn(7,7,3,3,1).astype(np.float32) * std
			self.pixel3s_b = np.zeros((7,7,1)).astype(np.float32)
			
			self.pixel4s_w = np.random.randn(7,7,3,3,1).astype(np.float32) * std
			self.pixel4s_b = np.zeros((7,7,1)).astype(np.float32)
			
			self.pixel5s_w = np.random.randn(7,7,3,3,1).astype(np.float32) * std
			self.pixel5s_b = np.zeros((7,7,1)).astype(np.float32)
			
			self.pixel6s_w = np.random.randn(7,7,3,3,1).astype(np.float32) * std
			self.pixel6s_b = np.zeros((7,7,1)).astype(np.float32)
			
			self.pixel7s_w = np.random.randn(1,7,3,3,1).astype(np.float32) * std
			self.pixel7s_b = np.zeros((1,7,1)).astype(np.float32)
			
			self.padded_a0 = []
			self.padded_a1 = [2]
			self.padded_a2 = [2]
			self.padded_a3 = [2]
			self.padded_a4 = [2]
			self.padded_a5 = [2]
			self.padded_a6 = [2]
			self.padded_a7 = [2]
			
			self.layer1_z1 = []
			self.layer2_z2 = []
			self.layer3_z3 = []
			self.layer4_z4 = []
			self.layer5_z5 = []
			self.layer6_z6 = []
			
			
			self.dw1 = [3]
			self.dw2 = [3]
			self.dw3 = [3]
			self.dw4 = [3]
			self.dw5 = [3]
			self.dw6 = [3]
			self.dw7 = [3]
						

	
	
def xman_show(xcheckpoint):
	
	x_image = cv2.imread(f'files/seven_pixels_training/17.jpg')	
	a0 = x_image/255
	a0 = a0[:,:,::-1]
		
	checkpoint = np.load(xcheckpoint)
	pixel1s_w = checkpoint['pixel1s_w']
	pixel2s_w = checkpoint['pixel2s_w']
	pixel3s_w = checkpoint['pixel3s_w']
	pixel4s_w = checkpoint['pixel4s_w']
	pixel5s_w = checkpoint['pixel5s_w']
	pixel6s_w = checkpoint['pixel6s_w']
	pixel7s_w = checkpoint['pixel7s_w']
	pixel1s_b = checkpoint['pixel1s_b']
	pixel2s_b = checkpoint['pixel2s_b']
	pixel3s_b = checkpoint['pixel3s_b']
	pixel4s_b = checkpoint['pixel4s_b']
	pixel5s_b = checkpoint['pixel5s_b']
	pixel6s_b = checkpoint['pixel6s_b']
	pixel7s_b = checkpoint['pixel7s_b']
	
	padded_a0 = checkpoint['padded_a0']
	padded_a1 = checkpoint['padded_a1']
	padded_a2 = checkpoint['padded_a2']
	padded_a3 = checkpoint['padded_a3']
	padded_a4 = checkpoint['padded_a4']
	padded_a5 = checkpoint['padded_a5']
	padded_a6 = checkpoint['padded_a6']
	padded_a7 = checkpoint['padded_a7']
	
	count  = checkpoint['countt']
	print(f'count: {count}')
	
	"""
	layer1 = []
	for i in range(7):
		net_sum = myconvolution(a0,pixel1s_w[i]) + pixel1s_b[i]
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)

		layer1.append(mypadding(net_sum,3))
	layer1 = np.array(layer1)
	
	print('layer1 calcultions completed')
	
	layer2 = []
	for i in range(7):
		net_sum = 0
		for y in range(7):
			df = myconvolution(layer1[y],pixel2s_w[i][y]) + pixel2s_b[i][y]
			net_sum = df + net_sum
		
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)
		
		layer2.append(mypadding(net_sum,3))
	layer2 = np.array(layer2)
	
	print('layer2 calculations completed ')
	
	layer3 = []
	for i in range(7):
		net_sum = 0
		for y in range(7):
			df = myconvolution(layer2[y],pixel3s_w[i][y]) + pixel3s_b[i][y]
			net_sum = df + net_sum
		
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)
		
		layer3.append(mypadding(net_sum,3))
	layer3 = np.array(layer3)
	
	print('layer3 calculations completed ')
	
	layer4 = []
	for i in range(7):
		net_sum = 0
		for y in range(7):
			df = myconvolution(layer3[y],pixel4s_w[i][y]) + pixel4s_b[i][y]
			net_sum = df + net_sum
		
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)
		
		layer4.append(mypadding(net_sum,3))
	layer4 = np.array(layer4)
	
	print('layer4 calculations completed :')
	
	layer5 = []
	for i in range(7):
		net_sum = 0
		for y in range(7):
			df = myconvolution(layer4[y],pixel5s_w[i][y]) + pixel5s_b[i][y]
			net_sum = df + net_sum
		
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)
		
		layer5.append(mypadding(net_sum,3))
	layer5 = np.array(layer5)
	
	print('layer5 calculations completed ')
	
	layer6 = []
	for i in range(7):
		net_sum = 0
		for y in range(7):
			df = myconvolution(layer5[y],pixel6s_w[i][y]) + pixel6s_b[i][y]
			net_sum = df + net_sum
		
		net_sum = np.where(net_sum<=0,0.0001*net_sum,net_sum)
		net_sum = np.where(net_sum>10,np.sqrt(90+net_sum),net_sum)
		
		layer6.append(mypadding(net_sum,3))
	layer6 = np.array(layer6)
	
	print('layer6 calculations completed ')
	
	layer7 = []
	net_sum = 0
	for y in range(7):
		df = myconvolution(layer6[y],pixel7s_w[0][y]) + pixel7s_b[0][y]
		net_sum = df + net_sum
	 
	net_sum = 1/(1+np.exp(-net_sum))
	layer7.append(mypadding(net_sum,3))
	layer7 = np.array(layer7)
	
	print('layer7 calculations completed :')
	print()
	"""
	
	"""
	print('a1:',np.min(layer1),np.max(layer1))
	print('a2:',np.min(layer2),np.max(layer2))
	print('a3:',np.min(layer3),np.max(layer3))
	print('a4:',np.min(layer4),np.max(layer4))
	print('a5:',np.min(layer5),np.max(layer5))
	print('a6:',np.min(layer6),np.max(layer6))
	print('a7:',np.min(layer7),np.max(layer7))
	"""
	print('a1:',np.min(padded_a1),np.max(padded_a1))
	print('a2:',np.min(padded_a2),np.max(padded_a2))
	print('a3:',np.min(padded_a3),np.max(padded_a3))
	print('a4:',np.min(padded_a4),np.max(padded_a4))
	print('a5:',np.min(padded_a5),np.max(padded_a5))
	print('a6:',np.min(padded_a6),np.max(padded_a6))
	print('a7:',np.min(padded_a7),np.max(padded_a7))
	
	
	#layers = [layer1,layer2,layer3,layer4,layer5,layer6]
	padded_layers = [padded_a1,padded_a2,padded_a3,padded_a4,padded_a5,padded_a6]
	
	
	for t in range(6):
		"""
		for s in range(7):
			plt.subplot(1,7,s+1)
			plt.imshow(layers[t][s])
			plt.title(f'layer{t+1}{s+1}')
		plt.show()
		"""
		for s in range(7):
			plt.subplot(1,7,s+1)
			plt.imshow(padded_layers[t][s])
			plt.title(f'layer{t+1}{s+1}')
		plt.show()
		
		
	"""
	plt.subplot(1,1,1)
	plt.imshow(layer7[0])
	plt.title('final')
	plt.show()	
	"""
	
	plt.subplot(1,2,1)
	plt.imshow(a0)
	plt.title('Original Image')

	
	plt.subplot(1,2,2)
	plt.imshow(padded_a7[0])
	plt.title(f'final predicted output_{count}')
	plt.show()	

	
	
		

#p1 = sevenpixels()
#p1.initiate()
#p1.real_update()

		
	



tf  = f'files/test_result/7pixels/xcheckpoint_137.npz'	
xman_show(tf)		

		
		
		
		
		
		
		
		
		
		
		
		

	
		





		
		
		
			
			
		
		
			
		
	
		
		 
		
		
			
	
		



