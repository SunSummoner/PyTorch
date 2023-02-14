import torch

x = torch.empty(1)
print(x)

x = torch.empty(3)
print(x)
#It is like a 1d vector with 3 elements

x = torch.empty(2, 3)
print(x)
#It is like a 2d vector with 3 elements

x = torch.empty(2, 2, 3)
print(x)
#It is like a 3d vector with 3 elements

x = torch.empty(2, 2, 2, 3)
print(x)
#It is like a 4d vector with 3 elements

x = torch.rand(2,2)
print(x)
#Giving random values

x = torch.zeros(2, 2)
print(x)
#Like numpy

x = torch.ones(2, 2)
print(x)
#this will put ones in all the items

print(x.dtype)
#Default datatype of x

x = torch.ones(2, 2, dtype=torch.int)
print(x.dtype)
#Giving x a parameter


x = torch.ones(2, 2, dtype=torch.float16)
print(x.dtype)
#Giving x a parameter

print(x.size())
#size is a function

x = torch.tensor([2.3,0.5])
#Constructing a tensor from data
print(x)

x=torch.rand(2,2)
y=torch.rand(2,2)
#Some basic operations
print(x)
print(y)

z = x + y
print(z)
z = torch.add(x, y)
print(z)
#Both will do the same thing

y.add_(x)
print(y)
#Inplace addition
#Every function in PyTorch that has a trailing underscore performs inplace operations.

x=torch.rand(2,2)
y=torch.rand(2,2)

z = x - y
print(z)
z = torch.sub(x, y)
print(z)
#Both will do the same thing

y.sub_(x)
print(y)
#Inplace subtraction
#Every function in PyTorch that has a trailing underscore performs inplace operations.

x=torch.rand(2,2)
y=torch.rand(2,2)

z = x * y
print(z)
z = torch.mul(x, y)
print(z)
#Both will do the same thing

y.mul_(x)
print(y)
#Inplace multiplication

x=torch.rand(2,2)
y=torch.rand(2,2)

z = x / y
print(z)
z = torch.div(x, y)
print(z)
#Both will do the same thing

y.div_(x)
print(y)
#Inplace division

x = torch.rand(5, 3)
print(x)
print(x[:, 0])
#Printing all the rows and only the column 0

print(x[1, :])
print(x[1, 1])

print(x[1, 1].item())
#When there is only one item in the tensor this'll print its actual value

x = torch.rand(4,4)
print(x)
#Reshaping a tensor

y = x.view(16)
print (y)
#The number of elements must remain the same

y = x.view(-1,8)
print ("y is", y)
#When we don't want to specify one dimension

print(y.size())

#Converting from numpy to a torch tensor and vice versa
a = torch.ones(5)
print(a)
b=a.numpy()
print(type(b))
print(b)

a.add_(1)
print(a)
print(b)
#They share the same memory location if the tensor is on the GPU instead of CPU

a = np.ones(5)
print(a)
b=torch.from_numpy(a)
print(type(b))
print(b)

a +=1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    #Creating a tensor on the GPU
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    #z.numpy():
    #This will show error because of this'll be on the GPU
    print("Cuda works")
    z = z.to("cpu")

x = torch.ones(5, requires_grad=True)
print(x)
#For calculating optimisation and gradiants
