import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım: {device}")

    # ──────────────────────────────────────────────
    # VERİ ARTIRMA (AUGMENTATION)
    # ──────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # Renk bağımlılığını azaltır — angry/fear/disgust için faydalı
        transforms.RandomGrayscale(p=0.1),
        # Blur robustness — az örnekli sınıflarda genellemeyi artırır
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Resmin bir kısmını kapatır, tek bölgeye odaklanmayı önler
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transforms)

    # ──────────────────────────────────────────────
    # SINIF DENGESİZLİĞİ KONTROLÜ
    # ──────────────────────────────────────────────
    class_counts = np.array([
        sum(1 for _, l in train_dataset.samples if l == i)
        for i in range(len(train_dataset.classes))
    ])
    print("\nSınıf dağılımı:")
    for cls, cnt in zip(train_dataset.classes, class_counts):
        print(f"  {cls}: {cnt} örnek")

    # ──────────────────────────────────────────────
    # WEIGHTED RANDOM SAMPLER
    # Az örnekli sınıfların h'te görülmesini garanti eder
    # ──────────────────────────────────────────────
    sample_weights = [
        1.0 / class_counts[label]
        for _, label in train_dataset.samples
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # sampler kullanılınca shuffle=True kaldırılmalı
    train_loader = DataLoader(
        train_dataset, batch_size=64,
        sampler=sampler,
        num_workers=0, pin_memory=True
    )

    # ──────────────────────────────────────────────
    # MODEL
    # ──────────────────────────────────────────────
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 7)
    model = model.to(device)

    # ──────────────────────────────────────────────
    # CLASS WEIGHTS — az örnekli sınıflara kayıpta daha fazla ağırlık ver
    # ──────────────────────────────────────────────
    class_weights = torch.tensor(
        1.0 / class_counts, dtype=torch.float
    ).to(device)
    class_weights = class_weights / class_weights.sum() * len(train_dataset.classes)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    # ──────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ──────────────────────────────────────────────
    epochs = 30
    optimizer = optim.AdamW([
        {'params': model.features.parameters(),    'lr': 1e-5},   # Yavaş (fine-tune)
        {'params': model.classifier.parameters(),  'lr': 1e-4},   # Hızlı (head)
    ], weight_decay=1e-2)

    # Cosine Annealing: öğrenme hızını pürüzsüz düşürür
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ──────────────────────────────────────────────
    # EĞİTİM DÖNGÜSÜ
    # ──────────────────────────────────────────────
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[1]['lr']:.6f}")
        print("-" * 30)

        model.train()
        running_loss  = 0.0
        correct_train = 0
        total_train   = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping: gradyan patlamasını önler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss  += loss.item()
            _, predicted   = torch.max(outputs.data, 1)
            total_train   += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()

        train_accuracy = 100 * correct_train / total_train
        print(f"Eğitim -> Loss: {running_loss/len(train_loader):.4f} | Accuracy: %{train_accuracy:.2f}")

        if train_accuracy > best_acc:
            best_acc = train_accuracy
            torch.save(model.state_dict(), 'best_emotion_mobilenetv3.pth')
            print(f"  ✓ Yeni en iyi model kaydedildi (%{best_acc:.2f})")

    print(f"\nEğitim bitti! En yüksek eğitim başarısı: %{best_acc:.2f}")

if __name__ == '__main__':
    main()