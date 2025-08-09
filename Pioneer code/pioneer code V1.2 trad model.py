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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ImageEnhance
import math
import time
targetStyles = {
    'Baroque': 4,
    'Romanticism': 23,
    'Realism': 21,
    'Impressionism': 12,
    'Post_Impressionism': 20,
    'Expressionism': 9
}

auger_config = {
    'do_augment': True,
    'augmentations_per_image': 1,          # increase to expand training set size
    'max_rotation_deg': 20,                # do not rotate beyond this absolute value
    'allow_horizontal_flip': True,         # horizontal flips only
    'allow_vertical_flip': False,          # keep paintings upright
    'brightness_jitter': 0.15,             # +/- 15%
    'contrast_jitter': 0.15                # +/- 15%
}

feature_toggles = {
    'color_hist': True,
    'color_stats': True,
    'edge_texture': True,
    'lbp': True,
    'glcm': True,
    'texture_energy': True,
    'gist': True,
    'orb': True
}

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 1200  # ensure ~840 train per class (70%), >=1000 total easily; tune as needed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def extractGLCM(img):
    gray_img = rgb2gray(img)
    # Quantize to 16 gray levels
    q = np.clip((gray_img * 16).astype(np.uint8), 0, 15)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy']
    feats = []
    for p in props:
        vals = graycoprops(glcm, prop=p)  # shape (len(distances), len(angles))
        feats.extend(vals.flatten().tolist())
    return feats  # 2 distances x 4 angles x 3 props = 24 dims

def extractORBFeatures(img):
    gray_img = rgb2gray(img)
    gray_uint8 = (gray_img * 255).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray_uint8, None)
    num_kp = 0 if keypoints is None else len(keypoints)
    if num_kp == 0 or descriptors is None:
        # 69-dim zero vector placeholder
        return [0.0] * 69
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.angle for kp in keypoints if kp.angle >= 0], dtype=np.float32)
    # If some angles are -1 (unknown), handle gracefully
    if angles.size == 0:
        mean_cos, mean_sin = 0.0, 0.0
    else:
        radians = np.deg2rad(angles)
        mean_cos = float(np.mean(np.cos(radians)))
        mean_sin = float(np.mean(np.sin(radians)))
    # Descriptor stats (uint8, shape [N, 32])
    desc = descriptors.astype(np.float32) / 255.0
    desc_mean = np.mean(desc, axis=0).tolist()          # 32
    desc_std = np.std(desc, axis=0).tolist()            # 32
    feats = [
        float(num_kp),
        float(np.mean(responses)), float(np.std(responses)),
        float(np.mean(sizes)), float(np.std(sizes)),
        mean_cos, mean_sin
    ]
    feats.extend(desc_mean)
    feats.extend(desc_std)
    return feats  # total 69 dims

def get_feature_group_lengths():
    lengths = {}
    if feature_toggles['color_hist']:
        lengths['Color Histogram'] = 24
    if feature_toggles['color_stats']:
        lengths['Color Statistics'] = 12
    if feature_toggles['edge_texture']:
        lengths['Edge Texture'] = 4
    if feature_toggles['lbp']:
        lengths['LBP Texture'] = 26
    if feature_toggles['glcm']:
        lengths['GLCM Texture'] = 24
    if feature_toggles['texture_energy']:
        lengths['Texture Energy'] = 4
    if feature_toggles['gist']:
        lengths['GIST Spatial'] = 64
    if feature_toggles['orb']:
        lengths['ORB Keypoints'] = 69
    return lengths

def get_feature_group_indices():
    lengths = get_feature_group_lengths()
    indices = {}
    start = 0
    for name, length in lengths.items():
        indices[name] = (start, start + length)
        start += length
    return indices

def expected_feature_dim():
    return sum(get_feature_group_lengths().values())

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
    feats = []
    if feature_toggles['color_hist']:
        feats.extend(extractColorHist(img))
    if feature_toggles['color_stats']:
        feats.extend(extractColorStats(img))
    if feature_toggles['edge_texture']:
        feats.extend(extractEdgeTexture(img))
    if feature_toggles['lbp']:
        feats.extend(extractLBP(img))
    if feature_toggles['glcm']:
        feats.extend(extractGLCM(img))
    if feature_toggles['texture_energy']:
        feats.extend(extractTextureEnergy(img))
    if feature_toggles['gist']:
        feats.extend(extractGIST(img))
    if feature_toggles['orb']:
        feats.extend(extractORBFeatures(img))
    return np.array(feats, dtype=np.float32)

def augment_image(pil_img):
    img = pil_img
    # Rotation (bounded)
    max_deg = max(0, int(auger_config['max_rotation_deg']))
    if max_deg > 0:
        deg = random.uniform(-max_deg, max_deg)
        img = img.rotate(deg, resample=Image.BILINEAR, expand=True, fillcolor=(int(255*0.5),)*3)
        img = img.resize((256, 256), Image.LANCZOS)
    # Horizontal flip only
    if auger_config['allow_horizontal_flip'] and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Vertical flip not allowed by default
    if auger_config['allow_vertical_flip'] and random.random() < 0.1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Brightness/Contrast jitter
    if auger_config['brightness_jitter'] > 0:
        b = 1.0 + random.uniform(-auger_config['brightness_jitter'], auger_config['brightness_jitter'])
        img = ImageEnhance.Brightness(img).enhance(b)
    if auger_config['contrast_jitter'] > 0:
        c = 1.0 + random.uniform(-auger_config['contrast_jitter'], auger_config['contrast_jitter'])
        img = ImageEnhance.Contrast(img).enhance(c)
    return img

def augment_training_set(train_items):
    if not auger_config['do_augment'] or auger_config['augmentations_per_image'] <= 0:
        return []
    os.makedirs('/tmp/art_cache_traditional/aug', exist_ok=True)
    augmented = []
    for item in train_items:
        src_path = item['imgPath']
        try:
            base_img = Image.open(src_path).convert('RGB')
        except Exception:
            continue
        for k in range(auger_config['augmentations_per_image']):
            aug_img = augment_image(base_img)
            aug_path = src_path.replace('/tmp/art_cache_traditional/', '/tmp/art_cache_traditional/aug/')
            base_name = os.path.splitext(os.path.basename(aug_path))[0]
            save_path = f"/tmp/art_cache_traditional/aug/{base_name}_aug{k}.png"
            aug_img.save(save_path, 'PNG')
            augmented.append({
                'imgPath': save_path,
                'style': item['style'],
                'styleId': item['styleId']
            })
    print(f"Augmented training samples added: {len(augmented)}")
    return augmented

allData = loadData(samplesPerClass=SAMPLES_PER_CLASS)
trainData, valData, testData = splitData(allData)
print(f"Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
augmented = augment_training_set(trainData)
if augmented:
    trainData = trainData + augmented
    print(f"After augmentation -> Train: {len(trainData)}")
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
print(f"Feature vector size: {xAll.shape[1]} dimensions (expected {expected_feature_dim()})")
xAll = np.nan_to_num(xAll, nan=0.0, posinf=1.0, neginf=-1.0)

xTrain = xAll[:len(trainData)]
yTrain = yAll[:len(trainData)]
xVal = xAll[len(trainData):len(trainData)+len(valData)]
yVal = yAll[len(trainData):len(trainData)+len(valData)]
xTest = xAll[len(trainData)+len(valData):]
yTest = yAll[len(trainData)+len(valData):]
trainFeatureMean = np.mean(xTrain, axis=0)
trainFeatureStd = np.std(xTrain, axis=0) + 1e-8
print("Normalizing features...")
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)
xTestScaled = scaler.transform(xTest)
print("\nTraining ML models...")
print("Fitting Random Forest...")
rf_start = time.time()
rfModel = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, verbose=1)
rfModel.fit(xTrainScaled, yTrain)
print(f"RF fit time: {time.time()-rf_start:.1f}s")
print("Fitting SVM...")
svm_start = time.time()
svmModel = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, verbose=True)
svmModel.fit(xTrainScaled, yTrain)
print(f"SVM fit time: {time.time()-svm_start:.1f}s")
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
    group_indices = get_feature_group_indices()
    groupImportance = {}
    for groupName, (start, end) in group_indices.items():
        groupImportance[groupName] = float(np.mean(featImportance[start:end]))
    sortedImportance = sorted(groupImportance.items(), key=lambda x: x[1], reverse=True)
    for featGroup, importance in sortedImportance:
        print(f"{featGroup:>18}: {importance:.4f}")
    # Top individual features
    print("\nTop 10 individual features by importance:")
    featImportance = rfModel.feature_importances_
    top_idx = np.argsort(featImportance)[-10:][::-1]
    idx_to_group = {}
    for groupName, (start, end) in group_indices.items():
        for i in range(start, end):
            idx_to_group[i] = groupName
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank:2d}. idx {idx:4d} [{idx_to_group.get(idx, 'Unknown')}]: {featImportance[idx]:.5f}")

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

# Confusion matrices for validation and test
print("\nConfusion Matrices:")
valPreds_rf = rfModel.predict(xValScaled)
valPreds_svm = svmModel.predict(xValScaled)
val_best = valPreds_rf if bestName == 'Random Forest' else valPreds_svm
test_best = bestPreds
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_val = confusion_matrix(yVal, val_best)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=styleNames)
disp_val.plot(ax=axes[0], cmap='Blues', colorbar=False, xticks_rotation=45)
axes[0].set_title('Validation Confusion Matrix')
cm_test = confusion_matrix(yTest, test_best)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=styleNames)
disp_test.plot(ax=axes[1], cmap='Greens', colorbar=False, xticks_rotation=45)
axes[1].set_title('Test Confusion Matrix')
plt.tight_layout()
plt.show()

# Per-image explanations for Random Forest using SHAP (if available)
def build_feature_name_list():
    names = []
    # Color Histogram (HSV 8 bins)
    if feature_toggles['color_hist']:
        channels = ['H', 'S', 'V']
        for ch in channels:
            for b in range(8):
                lo = b/8.0
                hi = (b+1)/8.0
                names.append(f"HSV-{ch} hist bin {b} [{lo:.2f}-{hi:.2f}]")
    # Color Stats
    if feature_toggles['color_stats']:
        channels = ['H', 'S', 'V']
        stats = ['mean', 'var', 'skew', 'kurt']
        for ch in channels:
            for st in stats:
                names.append(f"HSV-{ch} {st}")
    # Edge Texture (Canny edge ratios at thresholds)
    if feature_toggles['edge_texture']:
        thresholds = [0.2, 0.3, 0.4, 0.6]
        for t in thresholds:
            names.append(f"Canny edge ratio @{t}")
    # LBP
    if feature_toggles['lbp']:
        for k in range(26):
            names.append(f"LBP uniform bin {k}")
    # GLCM
    if feature_toggles['glcm']:
        distances = [1, 3]
        angles_deg = [0, 45, 90, 135]
        props = ['contrast', 'homogeneity', 'energy']
        for p in props:
            for d in distances:
                for a in angles_deg:
                    names.append(f"GLCM {p} d={d} a={a}°")
    # Texture Energy
    if feature_toggles['texture_energy']:
        names.extend([
            'GradMag mean', 'GradMag std', 'abs(gradX) mean', 'abs(gradY) mean'
        ])
    # GIST (4x4 blocks x 4 orientations)
    if feature_toggles['gist']:
        orientations = ['0°', '45°', '90°', '135°']
        for i in range(4):
            for j in range(4):
                for o in orientations:
                    names.append(f"GIST block({i},{j}) {o}")
    # ORB
    if feature_toggles['orb']:
        names.extend([
            'ORB num_kp', 'ORB response mean', 'ORB response std',
            'ORB size mean', 'ORB size std', 'ORB angle cos', 'ORB angle sin'
        ])
        for i in range(32):
            names.append(f"ORB desc_mean[{i}]")
        for i in range(32):
            names.append(f"ORB desc_std[{i}]")
    return names

def _format_feature_explanation(name: str, value: float, zscore: float, contribution: float) -> str:
    trend = "high" if zscore > 0.8 else ("low" if zscore < -0.8 else "typical")
    sign = "supports" if contribution > 0 else "opposes"
    if name.startswith("HSV-") and "hist bin" in name:
        return f"{trend} presence in {name.split(' ')[0]} {name.split('hist')[0].strip().split('-')[1]} range ({name.split('[')[1][:-1]}), {sign} class"
    if name.startswith("HSV-") and any(s in name for s in ["mean","var","skew","kurt"]):
        return f"{trend} {name}, {sign} class"
    if name.startswith("Canny edge ratio"):
        return f"{trend} edge density {name.split('@')[1]}, {sign} class"
    if name.startswith("LBP uniform bin"):
        return f"{trend} local texture pattern {name}, {sign} class"
    if name.startswith("GLCM"):
        return f"{trend} {name} (co-occurrence texture), {sign} class"
    if name in [
        'GradMag mean', 'GradMag std', 'abs(gradX) mean', 'abs(gradY) mean']:
        return f"{trend} gradient {name}, {sign} class"
    if name.startswith("GIST block"):
        return f"{trend} spatial response {name}, {sign} class"
    if name.startswith("ORB "):
        return f"{trend} keypoint/descriptor stat {name}, {sign} class"
    return f"{trend} {name}, {sign} class"

def explain_per_image_random_forest(model, x_scaled, x_raw, class_names, feat_mean, feat_std, top_k=5, sample_count=6):
    try:
        import shap
    except Exception as e:
        print("SHAP not available; to enable per-image explanations, install it: pip install shap")
        return
    print("\nPer-image explanations (Random Forest):")
    np.random.seed(RANDOM_SEED)
    idxs = np.random.choice(len(x_scaled), size=min(sample_count, len(x_scaled)), replace=False)
    explainer = shap.TreeExplainer(model)
    # shap_values is a list (one array per class): [n_classes x (n_samples, n_features)]
    shap_values = explainer.shap_values(x_scaled[idxs])
    feature_names = build_feature_name_list()
    for k, idx in enumerate(idxs):
        x_row_scaled = x_scaled[idx:idx+1]
        x_row_raw = x_raw[idx]
        pred = model.predict(x_row_scaled)[0]
        contrib = shap_values[pred][k] if isinstance(shap_values, list) else shap_values[k]
        # sort by absolute contribution
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        print(f"\nSample {k+1}: predicted -> {class_names[pred]}")
        simple_bullets = []
        for rank, fi in enumerate(order, 1):
            fname = feature_names[fi] if fi < len(feature_names) else f"feature[{fi}]"
            z = (x_row_raw[fi] - feat_mean[fi]) / feat_std[fi]
            phrase = _format_feature_explanation(fname, float(x_row_raw[fi]), float(z), float(contrib[fi]))
            print(f"  {rank}. {fname}: value={x_row_raw[fi]:.4f}, z={z:+.2f}, contribution={contrib[fi]:+.4f} -> {phrase}")
            # Store simplified bullet for narrative
            simple_bullets.append(phrase)
        print("  Note: positive contribution moves towards predicted class.")
        # Layperson-friendly narrative
        print("  In plain language: ")
        for b in simple_bullets:
            print(f"   - {b}.")
        print(f"   - Overall, these cues collectively suggest the painting resembles {class_names[pred]}.")

# Always run per-image explanations using Random Forest (for interpretability)
# Use validation set to avoid test leakage in metrics
explain_per_image_random_forest(rfModel, xValScaled, xVal, styleNames, trainFeatureMean, trainFeatureStd, top_k=6, sample_count=6)

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