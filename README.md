# Analýza provozu na kruhovém objezdu

Tento projekt poskytuje skript v jazyce Python pro analýzu videozáznamu z kruhového objezdu. Skript využívá detekci objektů pomocí YOLOv8 k identifikaci vozidel a následně počítá, kolik vozidel opustí kruhový objezd jednotlivými výjezdy.

## Struktura projektu

```
.
├── data/
│   └── roundabout.avi  (vstupní video)
├── output/
│   └── traffic_counts.csv (výstupní soubor s výsledky)
├── simple_car_detection.py (hlavní skript)
├── config.ini          (konfigurační soubor)
├── requirements.txt    (seznam závislostí)
└── README.md
```

## Instalace

1.  **Klonování repozitáře:**
    ```bash
    git clone <URL_repozitare>
    cd <nazev_repozitare>
    ```

2.  **Vytvoření a aktivace virtuálního prostředí:**
    Doporučuje se použít virtuální prostředí, aby se předešlo konfliktům mezi balíčky.
    ```bash
    python -m venv .venv
    ```
    Aktivace prostředí:
    -   Windows (PowerShell):
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    -   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Instalace závislostí:**
    Nainstalujte všechny potřebné knihovny pomocí přiloženého souboru `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Konfigurace

Veškerá nastavení skriptu se provádí v souboru `config.ini`.

-   `[general]`:
    -   `video_path`: Cesta k vstupnímu video souboru.
    -   `model_path`: Cesta k natrénovanému modelu YOLO (např. `yolov8n.pt`).
    -   `confidence_threshold`: Prahová hodnota spolehlivosti pro detekci (0.0 - 1.0).
    -   `visualize`: `true` pro zobrazení videa s detekcemi, `false` pro běh na pozadí.
    -   `output_csv`: Cesta pro uložení výsledného CSV souboru.

-   `[lines]`:
    -   Definice čar pro počítání na jednotlivých výjezdech.
    -   Formát: `exit_<ID> = x1,y1,x2,y2`, kde `(x1,y1)` a `(x2,y2)` jsou souřadnice dvou bodů definujících čáru.

## Použití

1.  **Umístění videa:**
    Ujistěte se, že váš video soubor (např. `roundabout.avi`) je umístěn ve složce `data/`.

2.  **Spuštění analýzy:**
    Spusťte hlavní skript.
    ```bash
    python simple_car_detection.py
    ```

3.  **Výsledky:**
    Po dokončení analýzy naleznete soubor `traffic_counts.csv` ve složce `output/`. Soubor obsahuje agregovaná data o počtu vozidel, která projela jednotlivými výjezdy, a to v minutových intervalech.

## Poznámky

-   Skript v současné podobě používá zjednodušenou logiku pro sledování a počítání vozidel. Pro produkční nasazení by bylo vhodné implementovat robustnější tracker (např. ByteTrack nebo DeepSORT) a přesnější geometrickou kontrolu protnutí čáry.
-   Výkon detekce závisí na kvalitě vstupního videa a zvoleném modelu YOLO.
