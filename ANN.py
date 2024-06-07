import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Định nghĩa mô hình ANN
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Phẳng hình ảnh thành một vector
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Kiểm tra thiết bị và sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải dữ liệu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Khởi tạo mô hình, hàm mất mát và bộ tối ưu
model = ANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm huấn luyện mô hình
def train_model(model, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Khởi tạo lại gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass và tối ưu hóa
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Huấn luyện {epoch+1}/{epochs}, Thất bại: {running_loss/len(trainloader):.4f}")
    print('Hoàn tất huấn luyện')

# Huấn luyện mô hình
train_model(model, trainloader, criterion, optimizer, epochs=100)

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), 'ann_model.pth')

# Đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Độ chính xác của mô hình trên 10000 hình ảnh thử nghiệm: {accuracy:.2f}%')

# Đánh giá mô hình
evaluate_model(model, testloader)
