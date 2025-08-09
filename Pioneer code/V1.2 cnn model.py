#!pip install datasets transformers torch torchvision matplotlib scikit-learn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random
from PIL import Image
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

targetStyles = {
    'Baroque': 4,
    'Romanticism': 23,
    'Realism': 21,
    'Impressionism': 12,
    'Post_Impressionism': 20,
    'Expressionism': 9
}

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
            if styleCounts[styleName] < samplesPerClass:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if img.width < 32 or img.height < 32:
                    continue
                img = img.resize((160, 160), Image.LANCZOS)
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
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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

allData = loadData(samplesPerClass=1500)
trainData, valData, testData = splitData(allData)
print(f"Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
trainImgs, trainLabels = getDataset(trainData, training=True)
valImgs, valLabels = getDataset(valData, training=False)
testImgs, testLabels = getDataset(testData, training=False)

batchSize = 16

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

model = models.resnet18(pretrained=True)
numFeatures = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(numFeatures, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 6)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def trainEpoch(model, dataLoader, criterion, optimizer):
    model.train()
    totalLoss = 0
    correct = 0
    total = 0
    for imgs, labels in dataLoader:
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
    avgLoss = totalLoss / len(dataLoader)
    accuracy = 100 * correct / total
    return avgLoss, accuracy

def validate(model, dataLoader, criterion):
    model.eval()
    totalLoss = 0
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
    accuracy = 100 * correct / total
    return avgLoss, accuracy, allPreds, allLabels

numEpochs = 40
trainLosses = []
trainAccs = []
valLosses = []
valAccs = []
bestAcc = 0

print("Starting training")
for epoch in range(numEpochs):
    print(f'\nEpoch {epoch+1}/{numEpochs}')
    trainLoader = createDataloader(trainImgs, trainLabels, shuffle=True)
    trainLoss, trainAcc = trainEpoch(model, trainLoader, criterion, optimizer)
    trainLosses.append(trainLoss)
    trainAccs.append(trainAcc)
    valLoss, valAcc, _, _ = validate(model, valLoader, criterion)
    valLosses.append(valLoss)
    valAccs.append(valAcc)
    print(f'Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.2f}%')
    print(f'Val Loss: {valLoss:.4f}, Val Acc: {valAcc:.2f}%')
    # to-do:save model


model.load_state_dict(torch.load('best_model.pth'))
testLoss, testAcc, testPreds, testLabels = validate(model, testLoader, criterion)
print(f"\nFinal Test Accuracy: {testAcc:.2f}%")
styleNames = list(targetStyles.keys())
print("\nClassification Report:")
print(classification_report(testLabels, testPreds, target_names=styleNames))

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
            transforms.Resize((224, 224)),
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