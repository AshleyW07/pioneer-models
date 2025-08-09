#!pip install datasets transformers torch torchvision matplotlib scikit-learn
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# Configurable settings
# -----------------------------
# Data
save_image_size = 256  # increase saved image resolution to avoid losing detail
input_size = 224       # network input size
batchSize = 16

# Augmentation controls (constrained & probabilistic)
augmentation_global_p = 0.6          # probability to apply each optional augmentation
rotation_degrees = 15                # do not rotate over 90; keep small per art priors
allow_horizontal_flip = True         # do not flip upside down; only horizontal optionally
color_jitter_strength = (0.2, 0.2, 0.2)

# Training controls
numEpochs = 40
max_batches_per_epoch = None         # e.g., set to 200 to observe mid-epoch quickly in Colab
log_every_n_batches = 50
validate_every_epoch = 1

# Checkpointing and early stopping
checkpoints_dir = 'checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)
save_checkpoint_every_n_epochs = 5
early_stopping_patience = 7          # None or integer epochs without val improvement

# Model controls
backbone_name = 'resnet18'           # options: resnet18/34/50, densenet121, vgg16, efficientnet_b0, custom_cnn
use_pretrained_backbone = True
classifier_hidden_dims = [512, 256]  # can try smaller: [256, 128] or single [256]
classifier_dropout_p = 0.5

# Internal dropout within ResNet blocks via hooks (experimental)
enable_internal_resnet_dropout = True
internal_resnet_dropout_p = 0.2

# Target classes
targetStyles = {
    'Baroque': 4,
    'Romanticism': 23,
    'Realism': 21,
    'Impressionism': 12,
    'Post_Impressionism': 20,
    'Expressionism': 9
}
styleNames = list(targetStyles.keys())


def loadData(samplesPerClass=1500):
    print("Loading WikiArt dataset")
    dataset = load_dataset("huggan/wikiart", streaming=True)
    trainStream = dataset['train']
    filteredItems = []
    styleCounts = {style: 0 for style in targetStyles.keys()}
    os.makedirs('/tmp/art_imgs', exist_ok=True)
    processedCount = 0
    for item in trainStream:
        processedCount += 1
        if processedCount % 500 == 0:
            print(f"Processed {processedCount} pictures...")
            totalCollected = sum(styleCounts.values())
            print(f"Collected {totalCollected}/{samplesPerClass * len(targetStyles)} target images")
        if item['style'] in targetStyles.values():
            styleName = None
            for k, v in targetStyles.items():
                if v == item['style']:
                    styleName = k
                    break
            if styleName is None:
                continue
            if styleCounts[styleName] < samplesPerClass:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if img.width < 32 or img.height < 32:
                    continue
                # Save at higher resolution to retain detail
                img = img.resize((save_image_size, save_image_size), Image.LANCZOS)
                path = f'/tmp/art_imgs/{styleName}_{styleCounts[styleName]}.jpg'
                img.save(path)
                styleId = list(targetStyles.keys()).index(styleName)
                filteredItems.append({
                    'imgPath': path,
                    'style': styleName,
                    'styleId': styleId
                })
                styleCounts[styleName] += 1
        if all(count >= samplesPerClass for count in styleCounts.values()):
            break
    print(f"\nFinal counts per style:")
    for style, count in styleCounts.items():
        print(f"{style}: {count}")
    print(f"Total loaded: {len(filteredItems)} images")
    return filteredItems


def getDataset(data, training=False):
    if training:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5 if allow_horizontal_flip else 0.0),
            transforms.RandomApply([transforms.RandomRotation(degrees=rotation_degrees)], p=augmentation_global_p),
            transforms.RandomApply([transforms.ColorJitter(brightness=color_jitter_strength[0],
                                                          contrast=color_jitter_strength[1],
                                                          saturation=color_jitter_strength[2])], p=augmentation_global_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    images = []
    labels = []
    for item in data:
        img = Image.open(item['imgPath']).convert('RGB')
        img = transform(img)
        images.append(img)
        labels.append(item['styleId'])
    return images, labels


def splitData(data):
    styleData = {}
    for item in data:
        style = item['style']
        if style not in styleData:
            styleData[style] = []
        styleData[style].append(item)
    trainData = []
    valData = []
    testData = []
    for style, items in styleData.items():
        random.shuffle(items)
        n = len(items)
        trainEnd = int(0.7 * n)
        valEnd = trainEnd + int(0.15 * n)
        trainData.extend(items[:trainEnd])
        valData.extend(items[trainEnd:valEnd])
        testData.extend(items[valEnd:])
    return trainData, valData, testData


# Load and split
allData = loadData(samplesPerClass=1500)
trainData, valData, testData = splitData(allData)
print(f"Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
trainImgs, trainLabels = getDataset(trainData, training=True)
valImgs, valLabels = getDataset(valData, training=False)
testImgs, testLabels = getDataset(testData, training=False)


def createDataloader(images, labels, shuffle=True):
    dataset = list(zip(images, labels))
    if shuffle:
        random.shuffle(dataset)
    batches = []
    for i in range(0, len(dataset), batchSize):
        batch = dataset[i:i+batchSize]
        if len(batch) == batchSize:
            imgBatch = torch.stack([item[0] for item in batch])
            labelBatch = torch.tensor([item[1] for item in batch])
            batches.append((imgBatch, labelBatch))
    return batches


trainLoader = createDataloader(trainImgs, trainLabels, shuffle=True)
valLoader = createDataloader(valImgs, valLabels, shuffle=False)
testLoader = createDataloader(testImgs, testLabels, shuffle=False)


def apply_internal_dropout_resnet(resnet_model: nn.Module, p: float = 0.2):
    """Apply dropout inside ResNet blocks via forward hooks (affects layer1-4)."""
    def hook_fn(module, inputs, output):
        return nn.functional.dropout(output, p=p, training=module.training)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(resnet_model, layer_name, None)
        if layer is None:
            continue
        for block in layer:
            block.register_forward_hook(hook_fn)


def create_model(backbone: str, num_classes: int, pretrained: bool = True,
                 hidden_dims=None, dropout_p: float = 0.5) -> nn.Module:
    if hidden_dims is None:
        hidden_dims = [512, 256]

    # -----------------
    # Custom small CNN
    # -----------------
    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    class SmallCNN(nn.Module):
        def __init__(self, num_classes: int, head_hidden_dims, head_dropout: float):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # 112x112

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # 56x56

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # 28x28
            )
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            classifier_layers = []
            prev_dim = 128
            for dim in head_hidden_dims:
                classifier_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(head_dropout)])
                prev_dim = dim
            classifier_layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)
            self.apply(initialize_weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Instantiate backbone
    if backbone == 'custom_cnn':
        model = SmallCNN(num_classes=num_classes, head_hidden_dims=hidden_dims, head_dropout=dropout_p)
        return model
    elif backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        numFeatures = model.fc.in_features
        classifier_in_dim = numFeatures
    elif backbone == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        classifier_in_dim = model.fc.in_features
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        classifier_in_dim = model.fc.in_features
    elif backbone == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        classifier_in_dim = model.classifier.in_features
    elif backbone == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        classifier_in_dim = model.classifier[-1].in_features
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        classifier_in_dim = model.classifier[-1].in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Build classifier head
    layers = []
    prev_dim = classifier_in_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout_p)])
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, num_classes))

    if backbone.startswith('resnet'):
        model.fc = nn.Sequential(*layers)
        if enable_internal_resnet_dropout:
            apply_internal_dropout_resnet(model, p=internal_resnet_dropout_p)
    elif backbone.startswith('densenet'):
        model.classifier = nn.Sequential(*layers)
    elif backbone.startswith('vgg'):
        # Replace the last layers of VGG classifier
        model.classifier = nn.Sequential(
            nn.Flatten(), *layers
        )
    elif backbone.startswith('efficientnet'):
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),  # keep original dropout and extend with our head
            *layers
        )

    return model


num_classes = len(styleNames)
model = create_model(
    backbone=backbone_name,
    num_classes=num_classes,
    pretrained=use_pretrained_backbone,
    hidden_dims=classifier_hidden_dims,
    dropout_p=classifier_dropout_p,
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def trainEpoch(model, dataLoader, criterion, optimizer, max_batches=None, log_every=50):
    model.train()
    totalLoss = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels) in enumerate(dataLoader, start=1):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if log_every and batch_idx % log_every == 0:
            running_loss = totalLoss / batch_idx
            running_acc = 100 * correct / total if total > 0 else 0
            print(f"  [Batch {batch_idx}] Loss: {running_loss:.4f}, Acc: {running_acc:.2f}%")

        if max_batches is not None and batch_idx >= max_batches:
            break

    processed_batches = min(len(dataLoader), max_batches) if max_batches is not None else len(dataLoader)
    avgLoss = totalLoss / processed_batches
    accuracy = 100 * correct / total if total > 0 else 0
    return avgLoss, accuracy


def validate(model, dataLoader, criterion):
    model.eval()
    totalLoss = 0.0
    correct = 0
    total = 0
    allPreds = []
    allLabels = []
    with torch.no_grad():
        for imgs, labels in dataLoader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            totalLoss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            allPreds.extend(predicted.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
    avgLoss = totalLoss / len(dataLoader)
    accuracy = 100 * correct / total if total > 0 else 0
    return avgLoss, accuracy, allPreds, allLabels


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.show()


trainLosses = []
trainAccs = []
valLosses = []
valAccs = []
bestAcc = 0.0
epochs_no_improve = 0

print("Starting training")
for epoch in range(numEpochs):
    print(f'\nEpoch {epoch+1}/{numEpochs}')
    trainLoader = createDataloader(trainImgs, trainLabels, shuffle=True)
    trainLoss, trainAcc = trainEpoch(
        model, trainLoader, criterion, optimizer,
        max_batches=max_batches_per_epoch, log_every=log_every_n_batches
    )
    trainLosses.append(trainLoss)
    trainAccs.append(trainAcc)

    if (epoch + 1) % validate_every_epoch == 0:
        valLoss, valAcc, valPreds, valTrue = validate(model, valLoader, criterion)
        valLosses.append(valLoss)
        valAccs.append(valAcc)
        print(f'Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.2f}%')
        print(f'Val   Loss: {valLoss:.4f}, Val   Acc: {valAcc:.2f}%')

        # Plot/print confusion matrix for validation each epoch
        plot_confusion(valTrue, valPreds, styleNames)

        # Checkpoint best model
        if valAcc > bestAcc:
            bestAcc = valAcc
            epochs_no_improve = 0
            best_path = os.path.join(checkpoints_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path} (Val Acc: {bestAcc:.2f}%)")
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if (epoch + 1) % save_checkpoint_every_n_epochs == 0:
            ckpt_path = os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': valAcc,
                'val_loss': valLoss
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Early stopping
        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement.")
            break

# Evaluate best model on test set if available
best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")
else:
    print("Best model checkpoint not found. Evaluating current model.")

testLoss, testAcc, testPreds, testTrue = validate(model, testLoader, criterion)
print(f"\nFinal Test Accuracy: {testAcc:.2f}%")
print("\nClassification Report:")
print(classification_report(testTrue, testPreds, target_names=styleNames))

# Confusion matrix on test
plot_confusion(testTrue, testPreds, styleNames)

# Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(trainLosses, label='Train Loss')
plt.plot(valLosses, label='Val Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(trainAccs, label='Train Accuracy')
plt.plot(valAccs, label='Val Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()


def showExamples(model, testData, numSamples=6):
    model.eval()
    indices = random.sample(range(len(testData)), numSamples)
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        item = testData[idx]
        img = Image.open(item['imgPath']).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        imgTensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(imgTensor)
            predIdx = torch.argmax(output, dim=1).item()
        predStyle = styleNames[predIdx]
        trueStyle = item['style']
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        color = 'green' if predStyle == trueStyle else 'red'
        plt.title(f"Pred: {predStyle}\nTrue: {trueStyle}", color=color)
    plt.tight_layout()
    plt.show()

showExamples(model, testData)

import shutil
shutil.rmtree('/tmp/art_imgs', ignore_errors=True)