import torch 
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = y.shape

print(n_samples, n_features)
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training : f(5) = {model(x_test).item():.3f}')

#Training 

learning_rate = 0.01
n_iterations = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epochs in range(n_iterations):
    #Prediction
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    #Gradients = backward pass
    l.backward()

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if epochs%10 == 0:
        [w,b] = model.parameters()
        print(f'Epoch = {epochs}; w = {w[0][0].item():.3f}; loss = {l:.8f}')

print(f'Prediction before training : f(5) = {model(x_test).item():.3f}')

# Output :
# 4 1
# Prediction before training : f(5) = -5.027
# Epoch = 0; w = -0.531; loss = 68.37201691
# Epoch = 10; w = 1.369; loss = 1.87078404
# Epoch = 20; w = 1.681; loss = 0.14431071
# Epoch = 30; w = 1.738; loss = 0.09406035
# Epoch = 40; w = 1.753; loss = 0.08750271
# Epoch = 50; w = 1.762; loss = 0.08238158
# Epoch = 60; w = 1.769; loss = 0.07758581
# Epoch = 70; w = 1.776; loss = 0.07306997
# Epoch = 80; w = 1.782; loss = 0.06881686
# Epoch = 90; w = 1.789; loss = 0.06481139
# Prediction before training : f(5) = 9.576 