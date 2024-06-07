import tkinter as tk
from tkinter import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np

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

# Tải mô hình đã huấn luyện
model = ANN()
model.load_state_dict(torch.load('ann_model.pth'))
model.eval()

# Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Hàm nhận dạng chữ số
def predict_digit(img):
    # Chuyển đổi hình ảnh sang định dạng tensor
    img = img.resize((28, 28)).convert('L')
    img = np.array(img, dtype=np.float32)
    img = 1.0 - img / 255.0  # Chuẩn hóa giá trị pixel
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    
    # Dự đoán chữ số
    output = model(img)
    probabilities = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3_values = probabilities[top3_idx]
    return top3_idx, top3_values

# Tạo giao diện vẽ
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title('Nhận dạng chữ số')
        self.geometry('400x580')
        self.configure(bg='#F0F0F0')

        self.canvas = Canvas(self, width=280, height=280, bg='white', bd=2, relief='sunken')
        self.canvas.pack(pady=20)
        
        self.label = Label(self, text='Vẽ chữ số ở đây', font=('Helvetica', 18), bg='#F0F0F0')
        self.label.pack(pady=10)
        
        self.classify_btn = Button(self, text='Dự đoán', command=self.classify_handwriting, font=('Helvetica', 12), bg='#4CAF50', fg='white', padx=10, pady=5)
        self.classify_btn.pack(pady=10)
        
        self.clear_btn = Button(self, text='Làm mới', command=self.clear_canvas, font=('Helvetica', 12), bg='#F44336', fg='white', padx=10, pady=5)
        self.clear_btn.pack(pady=10)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        
        self.image1 = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.line([x1, y1, x2, y2], fill='black', width=5)

    def classify_handwriting(self):
        top3_idx, top3_values = predict_digit(self.image1)
        results = '\n'.join([f'{idx}: {value * 100:.2f}%' for idx, value in zip(top3_idx, top3_values)])
        self.label.configure(text=f'Top 3 dự đoán:\n{results}', font=('Helvetica', 15))  
         
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image1 = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)

# Chạy ứng dụng
app = App()
app.mainloop()
