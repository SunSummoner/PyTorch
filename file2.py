import torch 
x = torch.randn(3, requires_grad = True)
#Argument is necessary for grad
print(x)
y=x+2

print(y)

z = y*y*2
print(z)
#z =z.mean()
print(z)

v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v) #grandient ie is the rate of change dz/dz
print(x.grad)
#Scalar value is needed
#gradient tensor
#Uses Jacobian matrix

#Preventing PyTorch from preventing tracking
# x.require_grad_(False)
#x.detach()
# with torch.no_grad():

#x.requires_grad_(False)
#print(x)
#y = x.detach()
#print(y)
#with torch.no_grad():
#y = x+2
#print(y)

#Example

weights = torch.ones(4, requires_grad=True)

"""for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)"""

"""for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)"""

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()
    #Stops from summing up with each iteration

#OR

optimizer = torch.optim.SGD(weights, lr=0.01)
#PyTorch inbuilt optimizer
#lr = learning rate

optimizer.step()
optimizer.zero_grad()