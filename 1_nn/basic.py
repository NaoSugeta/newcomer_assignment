import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Define model
class NeuralNetwork(nn.Module): #nn.Moduleの継承
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #バッチはそのままで平坦化
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), #(input, output)
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # 10 class
        )

    def forward(self, x): # 順伝播
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits # last output

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #SGD:確率的勾配降下法


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = train_loss / len(dataloader)
    accuracy = correct / size

    print(f"Train Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    return avg_loss, accuracy

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # eval mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # withはファイル操作を便利にする
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, accuracy

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # データの構造
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # これでtestになる
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# N : batch size
# C : channel
# H : height
# W : weght

# Create data loaders.
# datase : そのまま素
# dataloader : バッチ化、シャッフルなどやってくれる、便利屋
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader: # Xにimage(バッチごと), yにlabelが返る
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #SGD:確率的勾配降下法


train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_dataloader, model, loss_fn)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

print("Done!")
print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


plt.figure(figsize=(12, 5))

# Lossグラフ
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Accuracyグラフ
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
name = os.path.splitext(os.path.basename(__file__))[0]
plt.savefig(f"{name}_curve.png")
#plt.show()

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(1)  # 各画像の予測ラベル
        all_preds.extend(preds.cpu().numpy())   # CPUに戻して記録
        all_labels.extend(y.cpu().numpy())

# 混同行列の計算
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# 混同行列の可視化
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.savefig(f"{name}_matrix.png")
#plt.show()