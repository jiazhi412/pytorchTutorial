import torch
import torch.nn


x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return x * w


learning_rate = 0.001
iter = 1000

optimizer = torch.optim.SGD([w], lr=learning_rate)
# network = torch.nn.Linear()
loss_function = torch.nn.MSELoss()

print(f"Prediction before training: f(5) = {forward(5).item()}")

for epoch in range(iter):
    y_pred = forward(x)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 100 == 0:
        print("epoch", epoch + 1, ' : w = ', w, ' loss = ', loss)

 
print(f"Prediction after training: f(5) = {forward(5).item()}")