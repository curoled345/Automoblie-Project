import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import zipfile
import os

# ✅ 설정
zip_path = '/content/traffic_detection_2.v1i.tensorflow.zip'  # 압축 파일 경로
extract_dir = '/content/traffic_data'  # 압축 해제 폴더
batch_size = 16
num_epochs = 20
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 압축 해제
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ 데이터셋 (ImageFolder 형식)
train_dataset = datasets.ImageFolder(f"{extract_dir}/train_images", transform=transform)
val_dataset = datasets.ImageFolder(f"{extract_dir}/val_images", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ✅ 클래스 수 확인
num_classes = len(train_dataset.classes)
print("클래스 목록:", train_dataset.classes)

# ✅ 모델 정의
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes).to(device)

# ✅ 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ✅ 검증 함수
def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"🔍 Validation Accuracy: {acc:.2f}%")
    return acc

# ✅ 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")
    evaluate(model, val_loader)

# ✅ 모델 저장
torch.save(model.state_dict(), "custom_cnn.pth")
print("✅ 모델 저장 완료")
