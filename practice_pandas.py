import numpy
import pandas

myarray = numpy.array([1,2,3])
rownames = ['a', 'b', 'c']

myseries = pandas.Series(myarray, index=rownames)

print(myseries)

multi_dimensional_array = numpy.array([[1,2,3], [4,5,6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']

mydataframe = pandas.DataFrame(multi_dimensional_array, index=rownames, columns=colnames)
print(mydataframe)

print("method 1:")
print("one column: {}".format(mydataframe['one']))  
print("method 2:")
print("one column: {}".format(mydataframe.one))  