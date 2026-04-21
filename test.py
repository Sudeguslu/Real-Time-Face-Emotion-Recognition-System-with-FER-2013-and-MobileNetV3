# test.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım: {device}")

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(root='dataset/test', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=2, persistent_workers=True)

    # Modeli yükle
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 7)
    model.load_state_dict(torch.load('best_emotion_mobilenetv3.pth', map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f"Test -> Loss: {val_loss/len(val_loader):.4f} | Accuracy: %{val_accuracy:.2f}")

if __name__ == '__main__':
    main()