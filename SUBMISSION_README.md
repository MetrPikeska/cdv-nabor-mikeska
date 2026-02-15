# Detekce a počítání vozidel na kruhovém objezdu

## Zadání

Navrhněte a implementujte skript, který:
- Detekuje projíždějící automobily
- Spočítá počty vozidel, která vyjíždějí jednotlivými větvemi křižovatky
- Agreguje data po minutách

## Řešení

**Hlavní skript:** `01_detect_and_count_submission.py`

### Princip činnosti

1. **Detekce vozidel** - YOLOv8 model detekuje auta a náklaďáky v každém snímku
2. **Sledování** - Každé vozidlo má unikátní ID a je sledováno mezi snímky (tracking)
3. **Určení pozice** - Skript ověří, zda je vozidlo v ROI (kruhový objezd)
4. **Počítání výjezdů** - Když vozidlo protne linii výjezdu, inkrementuje se počítadlo
5. **Agregace** - Výsledky se slučují po minutách

### Výstup

CSV soubor `output/car_crossings.csv`:

```
minute,exit_1,exit_2,exit_3,exit_4
0,5,3,4,2
1,7,5,6,4
...
TOTAL,125,98,112,88
```

### Jak spustit

#### 1. Instalace závislostí

```bash
# Vytvoření virtuálního prostředí (doporučuje se)
python -m venv .venv

# Aktivace (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalace balíčků
pip install -r requirements.txt
```

#### 2. Příprava souborů

- Video: `data/roundabout.avi`
- Model: `yolov8m.pt` (automaticky se stáhne poprvé)
- Výstup: Bude vytvořen v `output/car_crossings.csv`

#### 3. Spuštění

```bash
python 01_detect_and_count_submission.py
```

### Konfigurační parametry (v kódu)

```python
# Zabudované ROI a výjezdní polygony
ROI_POLYGON_COORDS = [...]
EXIT_LINES_COORDS = {"exit_1": [...], "exit_2": [...], ...}

# Sensitivity
conf=0.30  # Práh spolehlivosti YOLO (0.0-1.0)
TRACK_TIMEOUT=30  # Kolik snímků čekat než se vozidlo smaže
MIN_TRACK_LENGTH=5  # Minimálně snímků pro validní detekci
SHOW_VIDEO=False  # True = zobrazit okno během zpracování
```

## Technické detaily

### Použité technologie

- **YOLOv8** - detekce vozidel (object detection)
- **DeepSORT-like tracking** - sledování vozidel (persisten=True)
- **Shapely** - geometrické výpočty (polygony, průsečíky)
- **OpenCV** - čtení videa, vizualizace
- **PyTorch** - GPU akcelerace

### Harwarové požadavky

- GPU: NVIDIA GTX 1060 3GB (nebo lepší)
- RAM: 8GB+ (minimálně 4GB pro CPU-only režim)
- Video: 1920x1280, 30 FPS (přibližně)

### Výkon

- GTX 1060: ~2-3 snímky za vteřinu
- CPU (i7): ~1 snímek za vteřinu

## Možná rozšíření use-case-u

1. **Detekce směru** - Určit, odkud vozidlo přijelo a kam odjelo (matrice OD)
2. **Klasifikace vozidel** - Rozlišit auta/náklaďáky/motocykly s různými tarify
3. **Analýza rychlosti** - Měřit průměrné rychlosti na jednotlivých výjezdech
4. **Detekce incidentů** - Dlouhá nehybnost vozidla (nehoda), reverzní jízda
5. **Analýza bezpečnosti** - Detekce cyklistů a chodců, jejich bezpečné průjezdy
6. **Predikce zatížení** - Machine Learning modely pro prognózu zátěže
7. **Optimalizace semaforu** - Dynamické řízení semaforů na základě počtů vozidel
8. **Heat-map pohybu** - Vizualizace, kde se vozidla pohybují nejčastěji
9. **Dálkový monitoring** - Integraci více kamer, centrální dashboard
10. **Analýza emisí** - Odhad emisí na základě počtů vozidel a typů

## Řešení potíží

### Video se nenačte
```
Chyba: Nelze otevřít video soubor data/roundabout.avi
```
→ Zkontrolujte, že soubor existuje a má správné kódování

### CUDA není dostupné
```
CUDA available: False
```
→ Skript bude používat CPU (pomalejší), ale stále bude fungovat

### Nedostatek paměti GPU
```
RuntimeError: CUDA out of memory
```
→ Zmenšit model na `yolov8n.pt` nebo zvýšit `conf` na 0.50

## Autor

Vytvořeno pro projekt analýzy provozu na kruhovém objezdu.
