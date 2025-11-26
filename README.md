# 🎭 Real-time Emotion Analyzer

MediaPipe kullanarak gerçek zamanlı yüz ifadesi analizi ve duygu tanıma sistemi. Web kamerasından gelen görüntüleri analiz ederek 8 farklı duyguyu tespit eder ve görselleştirir.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Gereksinimler](#-gereksinimler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Klavye Kısayolları](#-klavye-kısayolları)
- [Proje Yapısı](#-proje-yapısı)
- [Teknik Detaylar](#-teknik-detaylar)
- [Duygu Kategorileri](#-duygu-kategorileri)
- [Örnekler](#-örnekler)

## ✨ Özellikler

### 🎯 Temel Özellikler
- **Gerçek Zamanlı Duygu Tanıma**: Web kamerasından canlı video analizi
- **8 Duygu Kategorisi**: Mutlu, Üzgün, Kızgın, Şaşkın, Korku, İğrenme, Küçümseme, Nötr
- **Çoklu Yüz Desteği**: Aynı anda 3 yüze kadar tespit ve analiz
- **Güven Skorları**: Her duygu tespiti için güvenilirlik yüzdesi

### 📊 Görselleştirme
- **Renk Kodlu Çerçeveler**: Her duygu için farklı renk
- **Gerçek Zamanlı İstatistikler**: Duygu dağılım yüzdeleri
- **Çubuk Grafik**: Duygu yoğunluğu görselleştirmesi
- **FPS Göstergesi**: Performans takibi

### 💾 Veri Yönetimi
- **Ekran Görüntüsü**: Anlık görüntü kaydetme (S tuşu)
- **Video Kaydı**: Duygu analizini video olarak kaydetme (R tuşu)
- **CSV Export**: Tüm verileri CSV formatında kaydetme (C tuşu)
- **Otomatik Klasör Yapısı**: Screenshots, videolar ve veriler için ayrı klasörler

### 🔬 Gelişmiş Analiz
- **Göz Açıklığı Analizi (EAR)**: Göz kapağı pozisyonu tespiti
- **Ağız Açıklığı Analizi (MAR)**: Ağız şekli ve genişliği ölçümü
- **Kaş Pozisyonu**: Kaş-göz mesafesi ve eğim analizi
- **Ağız Asimetrisi**: Yüz ifadesi simetrisi kontrolü

## 🔧 Gereksinimler

### Sistem Gereksinimleri
- Python 3.7 veya üzeri
- Web kamerası
- macOS, Linux veya Windows

### Python Kütüphaneleri
Tüm gerekli kütüphaneler `requirements.txt` dosyasında listelenmiştir:

```
mediapipe==0.10.21
opencv-python==4.11.0.86
opencv-contrib-python==4.11.0.86
numpy==1.26.4
```

## 📦 Kurulum

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/ozlem-dilek/realtime-emotion-analyzer.git
cd realtime-emotion-analyzer
```

### 2. Sanal Ortam Oluşturun (Önerilir)

```bash
python3 -m venv venv
```

### 3. Sanal Ortamı Aktifleştirin

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### 5. Programı Çalıştırın

```bash
python main.py
```

## 🚀 Kullanım

### Temel Kullanım

1. Programı başlattığınızda web kamerası otomatik olarak açılır
2. Kameraya bakın ve farklı yüz ifadeleri deneyin
3. Ekranda tespit edilen duygu ve güven skorunu görürsünüz
4. İstatistikler sol üstte, çubuk grafik sağ üstte görüntülenir

### İlk Çalıştırma

Program ilk çalıştırıldığında otomatik olarak şu klasörler oluşturulur:
- `screenshots/` - Ekran görüntüleri
- `videolar/` - Video kayıtları
- `veriler/` - CSV veri dosyaları

## ⌨️ Klavye Kısayolları

| Tuş | İşlev | Açıklama |
|-----|-------|----------|
| **S** | Ekran Görüntüsü | Anlık ekran görüntüsü alır ve `screenshots/` klasörüne kaydeder |
| **R** | Video Kaydı | Video kaydını başlatır/durdurur. `videolar/` klasörüne MP4 formatında kaydeder |
| **C** | CSV Kaydet | Tüm duygu verilerini CSV formatında `veriler/` klasörüne kaydeder |
| **Q** | Çıkış | Programı kapatır |

### Video Kaydı Kullanımı
- İlk **R** tuşuna basış: Kayıt başlar (ekranda "KAYIT: ACIK" görünür)
- İkinci **R** tuşuna basış: Kayıt durur ve dosya kaydedilir

## 📁 Proje Yapısı

```
duygu_tanima/
│
├── main.py                 # Ana program dosyası
├── requirements.txt        # Python bağımlılıkları
├── README.md              # Bu dosya
│
├── screenshots/           # Ekran görüntüleri (otomatik oluşturulur)
│   └── ekran_goruntusu_*.jpg
│
├── videolar/              # Video kayıtları (otomatik oluşturulur)
│   └── duygu_kaydi_*.mp4
│
└── veriler/               # CSV veri dosyaları (otomatik oluşturulur)
    └── duygu_verileri_*.csv
```

## 🔬 Teknik Detaylar

### Duygu Tanıma Algoritması

Program, MediaPipe Face Mesh kullanarak yüzdeki 468 landmark noktasını tespit eder ve şu özellikleri analiz eder:

1. **Göz Açıklığı (EAR - Eye Aspect Ratio)**
   - Göz kapağı pozisyonu
   - Göz açıklığı oranı

2. **Ağız Analizi (MAR - Mouth Aspect Ratio)**
   - Ağız genişliği
   - Ağız açıklığı
   - Ağız köşeleri pozisyonu

3. **Kaş Pozisyonu**
   - Kaş-göz mesafesi
   - Kaş eğimi

4. **Yüz Simetrisi**
   - Ağız köşeleri asimetrisi

### Skorlama Sistemi

Her duygu için ayrı skorlama yapılır:
- **Neutral**: Başlangıç bonusu (0.5) + normal değer aralıkları kontrolü
- **Diğer Duygular**: Belirgin yüz ifadesi özelliklerine göre skorlama
- En yüksek skorlu duygu seçilir
- Güven skoru yüzde olarak gösterilir

### Performans

- **FPS**: Gerçek zamanlı performans gösterimi
- **Çoklu Yüz**: Aynı anda 3 yüze kadar tespit
- **Optimizasyon**: Deque kullanarak bellek yönetimi

## 😊 Duygu Kategorileri

| Duygu | Türkçe | Renk | Özellikler |
|-------|--------|------|------------|
| **Happy** | Mutlu | 🟢 Yeşil | Geniş ağız, yukarı kıvrılmış köşeler, yükselmiş kaşlar |
| **Sad** | Üzgün | 🔵 Mavi | Aşağı kıvrılmış ağız, düşük kaşlar |
| **Angry** | Kızgın | 🔴 Kırmızı | Çatık kaşlar, dar gözler |
| **Surprised** | Şaşkın | 🟠 Turuncu | Açık gözler, açık ağız, yükselmiş kaşlar |
| **Fear** | Korku | 🟣 Mor | Açık gözler, açık ağız, düşük kaşlar |
| **Disgust** | İğrenme | 🟢 Koyu Yeşil | Dar ağız, dar gözler |
| **Contempt** | Küçümseme | ⚪ Gri | Asimetrik ağız, hafif yükselmiş köşe |
| **Neutral** | Nötr | ⚪ Beyaz | Normal yüz ifadesi, simetrik özellikler |

## 📊 CSV Veri Formatı

CSV dosyaları şu sütunları içerir:

```csv
Zaman,Duygu,Güven Skoru
2024-11-26 23:50:56.123,Neutral,0.75
2024-11-26 23:50:56.156,Happy,0.82
2024-11-26 23:50:56.189,Neutral,0.68
```

## 🎬 Örnekler

### Ekran Görüntüsü Alma
1. Programı çalıştırın
2. İstediğiniz bir anı yakalayın
3. **S** tuşuna basın
4. `screenshots/` klasöründe dosyayı bulun

### Video Kaydetme
1. Programı çalıştırın
2. **R** tuşuna basarak kaydı başlatın
3. Analiz yapın
4. Tekrar **R** tuşuna basarak kaydı durdurun
5. `videolar/` klasöründe MP4 dosyasını bulun

### Veri Analizi
1. Programı çalıştırın ve analiz yapın
2. **C** tuşuna basarak CSV kaydedin
3. Excel veya Python ile verileri analiz edin

## 🐛 Sorun Giderme

### Kamera Açılmıyor
- Kameranın başka bir program tarafından kullanılmadığından emin olun
- `cv2.VideoCapture(0)` değerini `cv2.VideoCapture(1)` olarak değiştirmeyi deneyin

### Düşük FPS
- Web kamerası çözünürlüğünü düşürün
- Başka programları kapatın

### Import Hataları
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Lisans

Bu proje eğitim amaçlıdır. İstediğiniz gibi kullanabilir ve değiştirebilirsiniz.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📧 İletişim

Sorularınız veya önerileriniz için issue açabilirsiniz.

---

**Not**: Bu proje MediaPipe ve OpenCV kullanarak geliştirilmiştir. Daha fazla bilgi için [MediaPipe Dokümantasyonu](https://google.github.io/mediapipe/) ve [OpenCV Dokümantasyonu](https://docs.opencv.org/) sayfalarını ziyaret edebilirsiniz.

