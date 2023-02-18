import torch


#Steps
#Prediction
#Gradients Computation using Autograd
#Loss Computation
#Parameter updates

# f = w * x 

# f = 2 * x

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w =torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction

def forward(x):

    return w * x

#loss = MSE
#Same as before

def loss(y, y_predicted):
    return((y_predicted - y)**2).mean()
#Same as before

#gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
#def gradient (x,y, y_predicted):
#return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3}')

#training

training_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    #gradient= bacward pass
    #dw = gradient(X,Y,y_pred)
    l.backward()
    #update weights
    with torch.no_grad():
        w -= training_rate *w.grad

    #zero grad
    w.grad.zero_()

    if epoch % 10 == 0:
        print (f'epoch{epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

    



