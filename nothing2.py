import os


image_folder = 'files/images/train'
tx = os.listdir(image_folder)
print(tx[0],'dhfbnd')
f = os.path.join(image_folder)
ff = os.listdir(f)

#print(type(ff),ff[1].name,type(ff[1]))

xx = sorted(ff)
for x in xx:
	print(x)

t = os.path.splitext(ff[3])
print(t)
print(t[0])