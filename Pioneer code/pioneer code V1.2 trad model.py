#!pip install datasets scikit-learn opencv-python scikit-image matplotlib numpy pillow
# this is copied from google colab code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray, rgb2hsv
import random
from PIL import Image
import os
targetStyles = {
    'Baroque': 4,
    'Romanticism': 23,
    'Realism': 21,
    'Impressionism': 12,
    'Post_Impressionism': 20,
    'Expressionism': 9
}

def loadData(samplesPerClass=1500):
    print("Loading WikiArt dataset...")
    dataset = load_dataset("huggan/wikiart", streaming=True)
    trainStream = dataset['train']
    filteredItems = []
    styleCounts = {style: 0 for style in targetStyles.keys()}
    os.makedirs('/tmp/art_cache_traditional', exist_ok=True)
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
                img = img.resize((256, 256), Image.LANCZOS)
                imgArray = np.array(img, dtype=np.float32) / 255.0
                # Gaussian kernel
                kernel = cv2.getGaussianKernel(5, 1.0)
                kernel = np.outer(kernel, kernel)
                for c in range(3):
                    imgArray[:,:,c] = cv2.filter2D(imgArray[:,:,c], -1, kernel)
                img = Image.fromarray((imgArray * 255).astype(np.uint8))
                path = f'/tmp/art_cache_traditional/{styleName}_{styleCounts[styleName]}.png'
                img.save(path, 'PNG')
                filteredItems.append({
                    'imgPath': path,
                    'style': styleName,
                    'styleId': list(targetStyles.keys()).index(styleName)
                })
                styleCounts[styleName] += 1
        if all(count >= samplesPerClass for count in styleCounts.values()):
            break
    print(f"\nFinal counts per style:")
    for style, count in styleCounts.items():
        print(f"{style}: {count}")
    return filteredItems

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

def extractColorHist(img):
    #HSV channels using 8-bin histograms
    hsvImg = rgb2hsv(img)
    features = []
    for ch in range(3):
        hist, _ = np.histogram(hsvImg[:,:,ch], bins=8, range=(0,1))
        hist = hist / (np.sum(hist) + 1e-7)
        features.extend(hist)
    return features

def extractColorStats(img):
    #mean, variance, skewness, kurtosis
    hsvImg = rgb2hsv(img)
    features = []
    for ch in range(3):
        chData = hsvImg[:,:,ch].flatten()
        mean = np.mean(chData)
        variance = np.var(chData)
        std = np.sqrt(variance)
        features.append(mean)
        features.append(variance)
        # Skewness& kurtosis
        skewness = np.mean(((chData - mean) / std) ** 3)
        kurtosis = np.mean(((chData - mean) / std) ** 4)
        features.append(skewness)
        features.append(kurtosis)
    return features

def extractEdgeTexture(img):
    #Canny thresholds
    grayImg = rgb2gray(img)
    grayUint8 = (grayImg * 255).astype(np.uint8)
    features = []
    # Multiple thresholds (0.2, 0.3, 0.4, 0.6)
    thresholds = [0.2, 0.3, 0.4, 0.6]
    for thresh in thresholds:
        lowThresh = int(thresh * 255 * 0.5)
        highThresh = int(thresh * 255)
        edges = cv2.Canny(grayUint8, lowThresh, highThresh)
        edgeRatio = np.sum(edges > 0) / edges.size
        features.append(edgeRatio)
    return features

def extractLBP(img):
    grayImg = rgb2gray(img)
    radius = 3
    nPoints = 8 * radius
    lbp = local_binary_pattern(grayImg, nPoints, radius, method='uniform')
    nBins = nPoints + 2  # uniform patterns + non-uniform
    hist, _ = np.histogram(lbp.ravel(), bins=nBins, range=(0, nBins))
    hist = hist / (np.sum(hist) + 1e-7)  #normalize
    return hist.tolist()

def extractGIST(img):
    #4x4 grid, orientation filters
    grayImg = rgb2gray(img)
    resized = cv2.resize((grayImg * 255).astype(np.uint8), (32, 32))
    features = []
    gridSize = 4
    blockSize = 32 // gridSize
    orientations = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),    # 0°
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32),    # 45°
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),    # 90°
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32)     # 135°
    ]
    for i in range(gridSize):
        for j in range(gridSize):
            y1, y2 = i * blockSize, (i + 1) * blockSize
            x1, x2 = j * blockSize, (j + 1) * blockSize
            block = resized[y1:y2, x1:x2]
            for orientation in orientations:
                response = cv2.filter2D(block.astype(np.float32), -1, orientation)
                features.append(np.mean(np.abs(response)))
    return features

def extractTextureEnergy(img):
    grayImg = rgb2gray(img)
    #Sobel gradients
    gradX = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=3)
    gradY = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=3)
    gradMag = np.sqrt(gradX**2 + gradY**2)
    features = [
        np.mean(gradMag),
        np.std(gradMag),
        np.mean(np.abs(gradX)),
        np.mean(np.abs(gradY))]
    return features

def extractAllFeatures(imgPath):
    img = np.array(Image.open(imgPath).convert('RGB'))
    img = img / 255.0
    allFeatures = []
    allFeatures.extend(extractColorHist(img))      #24 features (8binsx3channels)
    allFeatures.extend(extractColorStats(img))     #12 features (4 statsx3 channels)
    allFeatures.extend(extractEdgeTexture(img))    #4 features
    allFeatures.extend(extractLBP(img))        #26 features
    allFeatures.extend(extractTextureEnergy(img))   #4 features
    allFeatures.extend(extractGIST(img))        #64 features (4×4 gridx4 orientations)
    return np.array(allFeatures)

allData = loadData(samplesPerClass=1500)
trainData, valData, testData = splitData(allData)
print(f"Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
print("\nExtracting traditional CV features...")
allDataCombined = trainData + valData + testData
xAll = []
yAll = []

for i, item in enumerate(allDataCombined):
    if i % 100 == 0:
        print(f"Processing image {i+1}/{len(allDataCombined)}")
    features = extractAllFeatures(item['imgPath'])
    xAll.append(features)
    yAll.append(item['styleId'])

xAll = np.array(xAll)
yAll = np.array(yAll)
print(f"Feature vector size: {xAll.shape[1]} dimensions")
xAll = np.nan_to_num(xAll, nan=0.0, posinf=1.0, neginf=-1.0)

xTrain = xAll[:len(trainData)]
yTrain = yAll[:len(trainData)]
xVal = xAll[len(trainData):len(trainData)+len(valData)]
yVal = yAll[len(trainData):len(trainData)+len(valData)]
xTest = xAll[len(trainData)+len(valData):]
yTest = yAll[len(trainData)+len(valData):]
print("Normalizing features...")
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)
xTestScaled = scaler.transform(xTest)
print("\nTraining ML models...")
rfModel = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rfModel.fit(xTrainScaled, yTrain)
svmModel = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svmModel.fit(xTrainScaled, yTrain)
rfPreds = rfModel.predict(xTestScaled)
svmPreds = svmModel.predict(xTestScaled)
rfAcc = accuracy_score(yTest, rfPreds)
svmAcc = accuracy_score(yTest, svmPreds)

print(f"\nModel Performance:")
print(f"Random Forest Accuracy: {rfAcc:.3f}")
print(f"SVM Accuracy: {svmAcc:.3f}")
bestModel = rfModel if rfAcc > svmAcc else svmModel
bestPreds = rfPreds if rfAcc > svmAcc else svmPreds
bestName = "Random Forest" if rfAcc > svmAcc else "SVM"
styleNames = list(targetStyles.keys())
print(f"\nBest Model: {bestName}")
print("\nClassification Report:")
print(classification_report(yTest, bestPreds, target_names=styleNames, digits=3))

if bestName == "Random Forest":
    print("\nFeature Importance Analysis:")
    featImportance = rfModel.feature_importances_
    featGroups = {
        'Color Histogram': (0, 24),
        'Color Statistics': (24, 36),
        'Edge Texture': (36, 40),
        'LBP Texture': (40, 66),
        'GLCM Texture': (66, 90),
        'Texture Energy': (90, 94),
        'GIST Spatial': (94, 158)
    }
    groupImportance = {}
    for groupName, (start, end) in featGroups.items():
        groupImportance[groupName] = np.mean(featImportance[start:end])
    sortedImportance = sorted(groupImportance.items(), key=lambda x: x[1], reverse=True)
    for featGroup, importance in sortedImportance:
        print(f"{featGroup:>15}: {importance:.4f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
models = ['Random Forest', 'SVM']
accuracies = [rfAcc, svmAcc]
colors = ['lightblue', 'lightyellow']
bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', va='bottom')
if bestName == "Random Forest" and 'sortedImportance' in locals():
    plt.subplot(1, 2, 2)
    groups, importances = zip(*sortedImportance)
    bars = plt.bar(groups, importances, color='lightcoral', edgecolor='darkred')
    plt.title('Feature Group Importance')
    plt.ylabel('Average Importance')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
def showExamples(testData, xTestScaled, bestModel, numSamples=6):
    indices = random.sample(range(len(testData)), numSamples)
    classNames = list(targetStyles.keys())
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        features = xTestScaled[idx:idx+1]
        predLabel = bestModel.predict(features)[0]
        trueLabel = yTest[idx]
        imgPath = testData[idx]['imgPath']
        img = Image.open(imgPath).convert('RGB')
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        titleColor = 'green' if predLabel == trueLabel else 'red'
        plt.title(f"Pred: {classNames[predLabel]}\nTrue: {classNames[trueLabel]}",
                  fontsize=12, color=titleColor)
    plt.tight_layout()
    plt.show()
showExamples(testData, xTestScaled, bestModel)
print(f"\nFINAL RESULTS SUMMARY")
print(f"Best model: {bestName}")
print(f"Test accuracy: {max(rfAcc, svmAcc):.3f}")
print(f"Feature dimensions: {xTrain.shape[1]}")
import shutil
shutil.rmtree('/tmp/art_cache_traditional', ignore_errors=True)