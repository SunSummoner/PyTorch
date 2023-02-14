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







