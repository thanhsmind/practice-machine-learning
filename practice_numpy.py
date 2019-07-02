import numpy

mylist = [[1,2,3,], [3,4,5]]
myarray = numpy.array(mylist)

print(myarray)
print(myarray.shape)
print('First row: {}'.format(myarray[0]))
print('Last row {}'.format(myarray[-1]))
print('Specific raw and col: {}'.format(myarray[1,1]))
print('Whole col: {}'.format(myarray[:, 1]))