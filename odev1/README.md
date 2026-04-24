# YZM304 Derin Öğrenme Lab Dersi - 1. Proje Ödevi

## Introduction
Ad: Hasna Sena Kaymak
Numara: 22290684

Bu çalışmada `healthcare-dataset-stroke-data.csv` veri seti kullanılarak inme (`stroke`) tahmini için ikili sınıflandırma problemi ele alınmıştır. Çalışmanın temel amacı, ders/laboratuvar kapsamında ele alınan tek gizli katmanlı temel çok katmanlı algılayıcı (MLP) modelini veri üzerinde eğitmek, performansını değerlendirmek, overfitting-underfitting davranışını incelemek ve farklı iyileştirme denemeleri ile sonuçları karşılaştırmaktır.

Veri seti sağlık göstergeleri, demografik bilgiler ve yaşam biçimiyle ilişkili öznitelikleri içermektedir. Hedef değişken `stroke` olup:

- `0`: inme yok
- `1`: inme var

Veri seti dengesizdir. Toplam `5110` gözlem içinde:

- `4861` adet `stroke=0`
- `249` adet `stroke=1`

bulunmaktadır. Pozitif sınıf oranı yalnızca `%4.87` olduğu için bu çalışmada yalnızca `accuracy` metriğine bakmak yeterli görülmemiş, `precision`, `recall`, `F1-score` ve `ROC-AUC` de birlikte değerlendirilmiştir.

Bu proje ödevi kapsamında şu hedefler yerine getirilmiştir:

- veri analizi yapılması
- gerekli ön işleme adımlarının uygulanması
- temel 1 gizli katmanlı MLP modelinin eğitilmesi
- daha derin ve/veya regülarize modellerin denenmesi
- model davranışlarının validation ve test setleri üzerinde incelenmesi
- aynı mimarinin `scikit-learn MLPClassifier` ile tekrar yazılması
- karmaşıklık matrisi ve temel ölçüm metriklerinin sunulması

## Methods

### 1. Proje dosya yapısı

```text
stroke_dataset/
|- healthcare-dataset-stroke-data.csv
|- notebook.ipynb
|- build_notebook.py
|- requirements.txt
|- README.md
```

### 2. Kullanılan ortam ve kütüphaneler

Çalışma aşağıdaki sürümler ile hazırlanmıştır:

- Python `3.11.8`
- numpy `2.4.4`
- pandas `3.0.2`
- matplotlib `3.10.8`
- seaborn `0.13.2`
- scikit-learn `1.8.0`
- notebook `7.5.5`
- ipykernel `7.2.0`
- nbformat `5.10.4`

### 3. Çalışmanın yeniden üretilmesi

PowerShell üzerinde proje klasöründe aşağıdaki adımlar uygulanabilir:

```powershell
cd C:\Users\hsena\Downloads\stroke_dataset

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name stroke-env --display-name "Python (stroke-env)"
python -m jupyter notebook
```

Jupyter veya VS Code içinde notebook açıldıktan sonra kernel olarak `Python (stroke-env)` seçilmelidir. Daha sonra `notebook.ipynb` dosyası `Run All` ile çalıştırılabilir.

### 4. Veri ön işleme

Uygulanan veri ön işleme adımları aşağıdaki gibidir:

- `id` sütunu modele alınmamıştır.
- Sayısal değişkenler:
  - `age`
  - `avg_glucose_level`
  - `bmi`
- İkili değişkenler:
  - `hypertension`
  - `heart_disease`
- Kategorik değişkenler:
  - `gender`
  - `ever_married`
  - `work_type`
  - `Residence_type`
  - `smoking_status`

Ön işleme ayrıntıları:

- `bmi` sütunundaki eksik değerler `median` ile doldurulmuştur.
- Sayısal değişkenler `StandardScaler` ile standardize edilmiştir.
- Kategorik değişkenler `OneHotEncoder(handle_unknown="ignore")` ile dönüştürülmüştür.
- Ön işleme sonrasında toplam özellik sayısı `21` olmuştur.

### 5. Veri bölme stratejisi

Veri sınıf oranı korunacak şekilde ayrılmıştır (`stratified split`):

- `train`: `3270` örnek
- `validation`: `818` örnek
- `test`: `1022` örnek

Yaklaşık pozitif sınıf oranı tüm bölümlerde `%4.86-%4.89` aralığında korunmuştur.

### 6. Modelleme yaklaşımı

#### 6.1 Temel NumPy modeli

Laboratuvar mantığını temsil eden özel bir `BinaryMLP` sınıfı NumPy ile yazılmıştır.

Temel mimari:

- giriş katmanı: `21` özellik
- gizli katman: `16` nöron
- çıkış katmanı: `1` nöron

Temel aktivasyon ve eğitim tercihleri:

- gizli katman aktivasyonu: `ReLU`
- çıkış aktivasyonu: `Sigmoid`
- kayıp fonksiyonu: `Binary Cross Entropy`
- optimizasyon: `Mini-batch SGD`

#### 6.2 İyileştirilmiş NumPy modelleri

Temel modele ek olarak iki iyileştirme denenmiştir:

1. `numpy_weighted_1_hidden`
- mimari: `(16,)`
- amaç: sınıf dengesizliğini azaltmak için pozitif sınıf ağırlığı kullanmak

2. `numpy_deep_l2`
- mimari: `(32, 16)`
- amaç: daha derin ağ ve `L2` regülarizasyon ile performans değişimini gözlemek

#### 6.3 Scikit-learn ile aynı mimarinin tekrar yazılması

`scikit-learn MLPClassifier` ile aynı mimari tekrar kurulmuştur.

Bu bölümde temel NumPy modeli ile `MLPClassifier` mümkün olduğunca aynı koşullarda karşılaştırılmıştır:

- aynı veri ayrımı
- aynı mimari `(16,)`
- aynı öğrenme oranı
- aynı batch boyutu
- aynı epoch sayısı
- aynı Xavier başlangıç ağırlıkları
- aynı `SGD` mantığı
- `momentum=0.0`
- `nesterovs_momentum=False`
- `shuffle=False`

Notebook içinde başlangıç tahmin farkı da raporlanmıştır:

- `Başlangıç tahmin farkı (NumPy vs scikit-learn, aynı ağırlıklar): 0.0000000000`

Bu sonuç, iki implementasyonun temel deney koşullarında aynı başlangıçtan başlatıldığını göstermektedir.

### 7. Hiperparametreler ve başlangıç ayarları

#### 7.1 Ortak başlangıç ayarları

- `random seed = 42`
- veri bölme: `train_test_split(..., stratify=y, random_state=42)`
- ilk ayrım: `test_size=0.20`
- ikinci ayrım: `validation_size=0.20` (`train_full` içinden)
- temel mimariler için ortak başlangıç: `Xavier/Glorot uniform`

#### 7.2 Deney tablosu

| Deney | Mimari | Optimizasyon | Learning rate | Batch size | Epoch | Shuffle | Regülarizasyon |
|---|---|---:|---:|---:|---:|---|---|
| NumPy temel model | `(16,)` | Mini-batch SGD | `0.03` | `64` | `120` | `False` | Yok |
| scikit-learn temel model | `(16,)` | SGD (`momentum=0`) | `0.03` | `64` | `120` | `False` | Yok |
| NumPy ağırlıklı model | `(16,)` | Mini-batch SGD | `0.03` | `64` | `300` | `True` | Sınıf ağırlığı |
| NumPy derin model | `(32, 16)` | Mini-batch SGD | `0.02` | `64` | `350` | `True` | `L2 = 1e-3` |

#### 7.3 Ek ayarlar

Temel NumPy model:

- `loss = Binary Cross Entropy`
- `patience = None`
- `pos_weight = 1.0`

Ağırlıklı NumPy model:

- `loss = weighted Binary Cross Entropy`
- `pos_weight = 19.57`
- `patience = 35`

Derin NumPy model:

- `loss = weighted Binary Cross Entropy`
- `pos_weight = 19.57`
- `patience = 40`
- `L2 lambda = 0.001`

Threshold seçimi:

- validation set üzerinde `0.01` ile `0.99` aralığında `0.01` adımlarla tarama yapılmıştır.
- seçim ölçütü sırasıyla:
  1. en yüksek `validation F1-score`
  2. eşitlik halinde daha yüksek `validation recall`
  3. eşitlik halinde daha yüksek `validation ROC-AUC`

Bu yaklaşım ile model seçimi test setine bakılarak yapılmamıştır.

## Results

### 1. Veri analizi bulguları

Veri analizi sonucunda aşağıdaki gözlemler elde edilmiştir:

- yalnızca `bmi` sütununda eksik değer vardır (`201` adet, yaklaşık `%3.93`)
- hedef değişken ciddi biçimde dengesizdir
- yaş ve ortalama glukoz seviyesi arttıkça `stroke` riskinin arttığı gözlenmektedir
- sınıf dengesizliği nedeniyle varsayılan `0.50` threshold değeri ile pozitif sınıfın kaçırılması olasıdır

### 2. NumPy modellerinin sonuçları

| Model | Validation F1 | Validation Recall | Validation ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `numpy_baseline_1_hidden` | `0.2778` | `0.5000` | `0.8386` | `0.8826` | `0.2500` | `0.7000` | `0.3684` | `0.8402` |
| `numpy_weighted_1_hidden` | `0.2609` | `0.3750` | `0.8415` | `0.9061` | `0.2830` | `0.6000` | `0.3846` | `0.8398` |
| `numpy_deep_l2` | `0.2523` | `0.7000` | `0.8287` | `0.8082` | `0.1682` | `0.7400` | `0.2741` | `0.8349` |

### 3. Scikit-learn model sonucu

| Model | Validation F1 | Validation Recall | Validation ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sklearn_mlp_same_architecture` | `0.2778` | `0.5000` | `0.8386` | `0.8826` | `0.2500` | `0.7000` | `0.3684` | `0.8402` |

Temel NumPy model ile `scikit-learn` modelinin aynı koşullar altında aynı sonuca ulaşması beklenen davranıştır ve deney kurgusunun tutarlı olduğunu göstermektedir.

### 4. Temel modelde threshold etkisi

Temel model için varsayılan ve seçilen threshold karşılaştırması:

| Senaryo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Varsayılan threshold `0.50` | `0.9511` | `0.5000` | `0.0200` | `0.0385` | `0.8402` |
| Validation ile seçilen threshold `0.12` | `0.8826` | `0.2500` | `0.7000` | `0.3684` | `0.8402` |

Bu tablo, dengesiz veri setlerinde yalnızca varsayılan threshold ile çalışmanın pozitif sınıfı büyük ölçüde kaçırabildiğini göstermektedir.

### 5. Seçilen model

Validation performansına göre seçilen model:

- `numpy_baseline_1_hidden`
- seçilen threshold: `0.12`

Seçilen modelin test seti sonuçları:

- `Accuracy = 0.8826`
- `Precision = 0.2500`
- `Recall = 0.7000`
- `F1-score = 0.3684`
- `ROC-AUC = 0.8402`

Sınıflandırma raporu:

| Sınıf | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `0` | `0.98` | `0.89` | `0.94` | `972` |
| `1` | `0.25` | `0.70` | `0.37` | `50` |

Karmaşıklık matrisi:

```text
[[867, 105],
 [ 15,  35]]
```

## Discussion

Bu çalışmada istenen proje adımları tamamlanmıştır. Veri analizi, ön işleme, temel MLP eğitimi, farklı mimari/iyileştirme denemeleri, validation kullanımı, metrik analizi ve `scikit-learn` ile aynı mimarinin tekrar yazımı başarılı biçimde gerçekleştirilmiştir.

Sonuçlar gösteriyor ki:

- veri seti belirgin biçimde dengesizdir
- bu nedenle yalnızca `accuracy` metriğine bakmak yanıltıcıdır
- varsayılan `0.50` threshold, pozitif sınıfı neredeyse tamamen kaçırmaktadır
- validation tabanlı threshold seçimi ile `class 1` yakalama başarısı ciddi biçimde artmıştır

Her ne kadar `class 1` için `precision` değeri düşük görünse de bu durum veri dengesizliği nedeniyle beklenen bir sonuçtur. Bu ödevin amacı yalnızca mümkün olan en yüksek skoru bulmak değil, model kurma sürecini doğru kurmak, farklı deneyleri karşılaştırmak ve sonuçları teknik olarak savunulabilir biçimde yorumlamaktır. Bu açıdan çalışma ödev gereksinimlerini karşılamaktadır.

Çalışmanın sınırlılıkları:

- pozitif sınıf oranı düşüktür
- veri miktarı sınırlıdır
- ileri düzey dengeleme teknikleri (SMOTE vb.) uygulanmamıştır
- batch normalization, dropout veya alternatif loss fonksiyonları denenmemiştir

Gelecek çalışmalar için öneriler:

- `SMOTE`, undersampling veya focal loss gibi dengesiz veri tekniklerinin denenmesi
- daha fazla regularization tekniği kullanılması
- farklı öğrenme oranları ve batch boyutlarının taranması
- `PyTorch` ile aynı mimarinin yeniden kurulması
- `class 1` için precision-recall dengesini daha iyi optimize edecek threshold araştırmaları yapılması
