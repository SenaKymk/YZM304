# YZM304 Derin Öğrenme - Proje 2

Bu klasör, proje tesliminin düzenlenmiş halidir. Çalışmada veri seti olarak `SVHN` kullanıldı ve dört farklı yaklaşım karşılaştırıldı:

- temel CNN
- iyileştirilmiş CNN
- `ResNet18`
- CNN özellikleri + `LinearSVC` hibrit modeli

## Klasör Düzeni

- `01_svhn_cnn_karsilastirma.ipynb`
  İlk notebook. Üç CNN modeli bu dosyada eğitiliyor ve karşılaştırılıyor.
- `02_hibrit_cnn_ml.ipynb`
  İkinci notebook. CNN'den özellik çıkarılıp hibrit model kuruluyor.
- `data/`
  SVHN veri dosyaları burada tutuluyor.
- `artifacts/`
  Asıl çıktı klasörü. Rapor için gereken dosyalar burada toplandı.
- `exports/`
  Colab'dan indirilen ham kopyalar burada saklandı. Projeyi çalıştırmak için zorunlu değiller.

## `artifacts/` İçeriği

- `models/`
  Eğitilmiş model ağırlıkları
- `figures/`
  Eğitim eğrileri ve karmaşıklık matrisleri
- `features/`
  Hibrit model için üretilen `.npy` dosyaları
- `cnn_comparison_metrics.csv`
  İlk notebooktaki üç modelin sonuçları
- `hybrid_comparison_metrics.csv`
  Hibrit notebook sonuçları

## Çalıştırma Sırası

1. `01_svhn_cnn_karsilastirma.ipynb`
2. `02_hibrit_cnn_ml.ipynb`

Colab'da çalıştırılacaksa veri seti ayrıca yüklenmez. Notebook içinde `download=True` kullanıldığı için SVHN otomatik iner. Drive bağlanmadan da çalışır; bu durumda notebook sonunda bulunan indirme hücresi ile `artifacts/` klasörü zip olarak alınabilir.

## Kullanılan Yöntem

İlk notebookta iki tane elle yazılmış CNN kuruldu. İlk model olabildiğince sade bırakıldı. İkinci modelde aynı iskelet korunup `BatchNorm2d` ve `Dropout` eklendi. Üçüncü model olarak `ResNet18` kullanıldı. Küçük boyutlu SVHN görüntülerine daha uygun olması için ilk katman ve maxpool kısmı yeniden ayarlandı.

İkinci notebookta `ImprovedCNN` içinden 256 boyutlu özellik vektörleri çıkarıldı. Bu özellikler `.npy` dosyalarına kaydedildi ve daha sonra `LinearSVC` ile sınıflandırma yapıldı.

## Özellik Dosyaları

Üretilen dosyaların boyutları:

- `X_train.npy`: `(65932, 256)`
- `y_train.npy`: `(65932,)`
- `X_test.npy`: `(26032, 256)`
- `y_test.npy`: `(26032,)`

Bu yapı, her örnek için 256 boyutlu bir derin özellik temsili çıkarıldığını gösteriyor.

## Sonuç Özeti

### Üç CNN modelinin karşılaştırması

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| Model 1 - Baseline CNN | 0.3674 | 91.00% |
| Model 2 - Improved CNN | 0.2646 | 92.52% |
| Model 3 - ResNet18 | 0.2478 | 93.62% |

### Hibrit model karşılaştırması

| Model | Test Accuracy |
|---|---:|
| Tam CNN (`ImprovedCNN`) | 92.68% |
| Hibrit Model (`CNN Features + LinearSVC`) | 92.56% |

## Kısa Yorum

- En iyi genel sonuç `ResNet18` ile alındı.
- `BatchNorm` ve `Dropout` eklemek temel CNN'e göre belirgin bir iyileşme sağladı.
- Hibrit model, tam CNN'e oldukça yakın sonuç verdi.
- Özellik çıkarımı sonrası klasik makine öğrenmesi tarafı çalıştı ve `.npy` üretimi sorunsuz tamamlandı.

## Rapor İçin Kullanılabilecek Dosyalar

- `artifacts/figures/cnn_training_curves.png`
- `artifacts/figures/cnn_confusion_matrices.png`
- `artifacts/figures/hybrid_cnn_training_curves.png`
- `artifacts/figures/hybrid_confusion_matrices.png`
- `artifacts/cnn_comparison_metrics.csv`
- `artifacts/hybrid_comparison_metrics.csv`

## Kısa Teslim Notu

Rapor yazılırken özellikle şu karşılaştırmaların verilmesi yeterli olur:

- temel CNN ile iyileştirilmiş CNN arasındaki fark
- `ResNet18` neden daha iyi sonuç verdi
- hibrit modelin tam CNN'e ne kadar yaklaştığı
- karmaşıklık matrislerinde en çok karışan sınıflar

## Kaynaklar

- PyTorch: https://pytorch.org/docs/stable/index.html
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- SVHN Dataset: http://ufldl.stanford.edu/housenumbers/
- Scikit-learn: https://scikit-learn.org/stable/
