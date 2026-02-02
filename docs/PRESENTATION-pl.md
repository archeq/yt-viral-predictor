# YouTube Viral Video Predictor
## Projekt z przedmiotu: Uczenie Maszynowe

---

## 1. Opis Projektu

### Temat
**Predykcja wiralowości filmów na YouTube na podstawie miniaturek i tytułów**

### Cel
Stworzenie modelu uczenia maszynowego, który na podstawie miniaturki (obrazu) i tytułu (tekstu) przewiduje, czy film na YouTube ma potencjał stania się viralem.

### Kontekst
Twórcy treści na YouTube często zastanawiają się, jakie elementy sprawiają, że film staje się popularny. Ten projekt wykorzystuje deep learning do analizy dwóch kluczowych elementów widocznych przed kliknięciem:
- **Miniaturka** - pierwszy element wizualny przyciągający uwagę
- **Tytuł** - tekst zachęcający do obejrzenia

---

## 2. Opis Problemu

### Rodzaj zadania
Regresja znormalizowana lub soft ranking - model przewiduje ciągły wynik wiralowości w zakresie [0, 1], gdzie wartości bliskie 1 oznaczają wyższy potencjał viralowy.

### Definicja matematyczna

Niech:
- $x_{img} \in \mathbb{R}^{224 \times 224 \times 3}$ - miniaturka filmu (obraz RGB)
- $x_{txt} \in \mathbb{N}^{L}$ - tytuł filmu (sekwencja tokenów o długości L)
- $y \in \[0, 1\]$ - etykieta (0 = słaby wynik, 1 = viral)

Model $f_\theta$ przewiduje:
$$\hat{y} = f_\theta(x_{img}, x_{txt}) = \sigma(g(h_{img}(x_{img}) \oplus h_{txt}(x_{txt})))$$

gdzie:
- $h_{img}: \mathbb{R}^{224 \times 224 \times 3} \rightarrow \mathbb{R}^{2048}$ - ekstraktor cech wizualnych (ResNet50)
- $h_{txt}: \mathbb{N}^{L} \rightarrow \mathbb{R}^{768}$ - ekstraktor cech tekstowych (DistilBERT)
- $\oplus$ - konkatenacja wektorów
- $g: \mathbb{R}^{2816} \rightarrow \mathbb{R}$ - klasyfikator (MLP)
- $\sigma$ - funkcja sigmoidalna

### Metryka wiralowości (V-Score)

Do określenia, czy film jest viralem, używamy **logarytmicznego V-Score** - znormalizowanej miary wydajności filmu względem historycznej średniej kanału:

$$V_{score} = \frac{\log(views + 1) - \mu_{baseline}}{\sigma_{baseline}}$$

gdzie:
- $\mu_{baseline}$ - mediana logarytmu wyświetleń z ostatnich 30 filmów kanału
- $\sigma_{baseline}$ - odchylenie standardowe logarytmu wyświetleń

**Interpretacja:**
| V-Score | Ocena |
|---------|--------------|
| > 1.0 | Viral  |
| < -0.5 | Słaby wynik  |

---

## 3. Dane Wejściowe i Wyjściowe

### Źródło danych
- **YouTube Data API v3** - wyszukiwanie kanałów po niszach
- **yt-dlp** - pobieranie metadanych filmów (tytuły, wyświetlenia, miniaturki)

### Format danych

#### Pliki CSV (`data/raw/{channel_id}.csv`)
| Kolumna | Typ | Opis |
|---------|-----|------|
| Video ID | string | Unikalny identyfikator filmu |
| Title | string | Tytuł filmu |
| Current Views | int | Liczba wyświetleń |
| V-Score | float | Obliczony wskaźnik wiralowości |

#### Miniaturki (`data/raw/thumbnails/{video_id}.jpg`)
- Format: JPEG
- Rozdzielczość: różna (skalowana do 224x224 podczas treningu)

### Podział danych
- Dane są **balansowane** poprzez undersampling (równa liczba virali i słabych filmów)
- Podział: **90% trening, 10% test** (stratified split - zachowuje proporcje klas)

### Replikacja danych

```bash
# 1. Ustaw klucz API YouTube w pliku api.txt
echo "YOUR_API_KEY" > api.txt

# 2. Uruchom pobieranie danych
python main.py

# 3. Dane zostaną zapisane w:
#    - data/raw/*.csv (metadane)
#    - data/raw/thumbnails/*.jpg (miniaturki)
```

---

## 4. Opis Algorytmu

### Architektura modelu (Multimodal Fusion)

Model łączy dwie gałęzie przetwarzania:

```
┌─────────────────┐     ┌─────────────────┐
│   Miniaturka    │     │     Tytuł       │
│  (224×224×3)    │     │   (max 50 tok)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│    ResNet50     │     │   DistilBERT    │
│   (frozen)      │     │    (frozen)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    [2048 dim]              [768 dim]
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Concatenate │
              │  [2816 dim] │
              └──────┬──────┘
                     │
                     ▼
              ┌──────────────┐
              │ Linear(1024) │
              │  BatchNorm   │
              │    ReLU      │
              │ Dropout(0.5) │
              │ Linear(256)  │
              │  BatchNorm   │
              │    ReLU      │
              │ Dropout(0.3) │
              │  Linear(1)   │
              │   Sigmoid    │
              └──────┬───────┘
                     │
                     ▼
              [Probability]
```

### Komponenty modelu

#### 1. Gałąź wizualna (ResNet50)
- **Architektura**: ResNet50 z wagami ImageNet (zamrożony)
- **Wyjście**: wektor 2048 cech
- **Transformacje wejścia (trening)**:
  - Resize do 256×256, RandomCrop do 224×224
  - RandomHorizontalFlip, ColorJitter, RandomRotation
  - RandomGrayscale, RandomErasing
  - Normalizacja: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

#### 2. Gałąź tekstowa (DistilBERT)
- **Architektura**: DistilBERT-base-uncased (zamrożony)
- **Wyjście**: token [CLS] → wektor 768 cech
- **Tokenizacja**: max_length=50, padding, truncation

#### 3. Klasyfikator (MLP z BatchNorm)
```python
self.classifier = nn.Sequential(
    nn.Linear(2816, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
```

### Funkcja kosztu

**Binary Cross-Entropy Loss (BCE)**: 

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Uwaga**: Pomimo użycia BCE, model przewiduje **ciągły wynik wiralowości** (nie dyskretną klasę). BCE działa jako funkcja kosztu dla **regresji z sigmoidą**, karząc za odległość predykcji od znormalizowanego celu.

### Algorytm optymalizacji

**Adam Optimizer** z **ReduceLROnPlateau scheduler**:
- Automatycznie zmniejsza LR gdy model przestaje się poprawiać

**Hiperparametry**:
- Learning rate: $\alpha = 3 \times 10^{-4}$ (tylko klasyfikator)
- Weight decay: $1 \times 10^{-3}$ (regularyzacja L2)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- Batch size: 16
- Early stopping patience: 10 epok

### Techniki regularyzacji

1. **Frozen backbone** - ResNet50 i DistilBERT są zamrożone
2. **Dropout** - 0.5 po pierwszej warstwie, 0.3 po drugiej
3. **BatchNorm** - normalizacja między warstwami
4. **Data Augmentation** - augmentacja obrazów treningowych
5. **Weight decay** - regularyzacja L2

### Proces treningu

1. **Załadowanie danych** z CSV i miniaturek
2. **Filtrowanie** - usunięcie "średniaków" (tylko V-Score > 1.0 lub < -0.5)
3. **Balansowanie** - undersampling do równej liczby klas
4. **Podział** - 90% trening, 10% test (stratified)
5. **Forward pass** - obliczenie predykcji
6. **Backward pass** - propagacja gradientów (tylko klasyfikator)
7. **Early stopping** - zatrzymanie gdy accuracy nie rośnie przez 10 epok

### Uruchomienie projektu

```bash
# Instalacja zależności
pip install -r requirements.txt

# Pobieranie danych (opcjonalne - jeśli nie ma danych)
python main.py

# Trening modelu
python train.py

# Model zostanie zapisany do: data/models/viral_predictor.pth
```

---

## 5. Wyniki i Wnioski

### Metryki treningu (z Early Stopping)

| Epoka | Train Loss | Test Loss | Test Accuracy |
|-------|------------|-----------|---------------|
| 1 | 0.69 | 0.67 | 58.75% |
| 2 | 0.60 | 0.62 | 64.83% ★ Best |
| 3 | 0.44 | 0.65 | 67.40% |
| ... | ... | ... | ... |
| 7 | 0.11 | 0.97 | 69.43% |

**Early stopping** zatrzymał trening w epoce 7 (patience=5), najlepszy model z epoki 2.

### Końcowe wyniki

- **Test Loss**: 0.62
- **Test Accuracy**: ~65-70%

### Przykładowe predykcje

| Miniaturka | Tytuł | Predykcja |
|------------|-------|-----------|
| 🖼️ Jasne kolory, twarz | "SHOCKING Discovery..." | 78% (viral) |
| 🖼️ Ciemne, nudne | "Tutorial part 5" | 32% (nie-viral) |

### Analiza

**Czynniki wpływające na wiralowość:**
1. **Miniaturki**: jasne kolory, twarze z emocjami, duży tekst
2. **Tytuły**: słowa kluczowe ("SHOCKING", "NEW", liczby), emocjonalny język

**Ograniczenia:**
- Model nie uwzględnia treści samego filmu
- Zależność od specyfiki kanału (V-Score normalizuje, ale nisze są różne)
- Ograniczona ilość danych treningowych

### Wnioski

1. **Multimodalne podejście** (obraz + tekst) pozwala na analizę obu elementów
2. **V-Score** jako metryka jest bardziej sprawiedliwa niż surowe wyświetlenia
3. **Transfer learning z frozen backbone** zapobiega overfittingowi na małym zbiorze
4. **Data Augmentation** zwiększa efektywną ilość danych treningowych
5. **Early stopping** chroni przed przeuczeniem modelu

---

## Struktura projektu

```
yt-viral-predictor/
├── main.py                    # Główny skrypt pobierania danych
├── download.py                # Klasa DataDownloader (API + yt-dlp)
├── dataset.py                 # PyTorch Dataset
├── model.py                   # Architektura ViralPredictor
├── train.py                   # Skrypt treningowy
├── api.txt                    # Klucz YouTube API (nie commitować!)
├── requirements.txt           # Zależności Python
├── data/   
│   ├── raw/                   # Surowe dane (CSV + miniaturki)
│   └── models/                # Wytrenowane modele (.pth)
└── docs/                      # Dokumentacja projektu
    ├── V-SCORE.md             # Dokumentacja algorytmu V-Score
    └── PRESENTATION-pl.md     # Ten dokument
```

---

## Wymagane biblioteki

```
torch
torchvision
transformers
pandas
numpy
Pillow
yt-dlp
google-api-python-client
tqdm
```

---


Projekt wykonany w ramach przedmiotu **Uczenie Maszynowe** (UAM 2025/2026)
