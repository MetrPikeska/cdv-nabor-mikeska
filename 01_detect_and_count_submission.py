"""
Detekce a počítání vozidel na kruhovém objezdu
=============================================

Zadání:
-------
Skript detekuje projíždějící automobily a spočítá počty vozidel, která vyjíždějí 
jednotlivými větvemi křižovatky (kruhu). Výstupem je tabulka s počtem vozidel 
pro každou minutu a každý výjezd.

Výstup:
-------
CSV soubor s formatem:
    minute,exit_1,exit_2,exit_3,exit_4
    0,5,3,4,2
    1,7,5,6,4
    ...
    TOTAL,125,98,112,88

Jak spustit:
-----------
    python 01_detect_and_count_submission.py

Zajišťuje:
---------
- Automatickou detekci vozidel pomocí YOLOv8
- Sledování vozidel mezi snímky (tracking)
- Počítání průjezdů jednotlivými výjezdy (exit_1, exit_2, exit_3, exit_4)
- Agregaci dat po minutách
- Vizualizaci (je možné vypnout pro batch zpracování)

Možná rozšíření use-case-u:
---------------------------
1. Detekce směru jízdy - určit kam vozidlo jelo antes a kam po
2. Klasifikace vozidel - rozlišit auta, náklaďáky, motocykly
3. Analýza rychlosti - měřit průměrné rychlosti na jednotlivých větvích
4. Detekce cyklistů a chodců - bezpečnostní analýza
5. Řetězení (trajectory) - sledovat konkrétní vozidla hlubší
6. Analýza zatížení - když je nejrušnější dobu
7. Detekce nehod - dlouhé nehyby vozidel, kolize
8. Prognózování - predikce zátěže v budoucnu
9. Optimalizace světel - řídající se podle počtů vozidel
10. Heat-map - vizualizace kde se vozidla pohybují
"""

import cv2
from ultralytics import YOLO
import torch
import json
from shapely.geometry import Point, Polygon
import csv
import os
import numpy as np
from collections import defaultdict

print("CUDA dostupné:", torch.cuda.is_available())

# ============================================================================
# ZABUDOVANÉ KONFIGURAČNÍ DATA (ROI a výjezdy)
# ============================================================================

# ROI polygon - kruhový objezd (definován vertexy)
ROI_POLYGON_COORDS = [[280, 1071], [458, 841], [506, 726], [462, 588], [350, 437], 
                      [3, 367], [6, 244], [344, 295], [510, 293], [878, 177], 
                      [962, 125], [1031, 1], [1159, 6], [1077, 163], [1093, 233], 
                      [1328, 399], [1917, 494], [1919, 628], [1378, 570], [1215, 632], 
                      [869, 782], [768, 892], [673, 1069]]

# Výjezdní polygony - jednotlivé větve kruhu
EXIT_LINES_COORDS = {
    "exit_1": [[1217, 522], [1178, 546], [1189, 612], [1275, 582], [1381, 565], [1417, 508], [1356, 507], [1287, 511], [1221, 519]],
    "exit_2": [[1018, 147], [1002, 196], [1004, 234], [1035, 256], [1110, 261], [1080, 223], [1066, 183], [1072, 152], [1020, 145]],
    "exit_3": [[577, 285], [497, 304], [388, 296], [349, 335], [453, 343], [575, 327], [613, 277], [577, 285]],
    "exit_4": [[577, 704], [540, 828], [514, 892], [442, 886], [476, 816], [508, 741], [509, 662], [579, 703]]
}

# ============================================================================
# DETEKČNÍ FUNKCE
# ============================================================================

def detect_cars(video_path, model_path, output_csv):
    """
    Hlavní funkce pro detekci a počítání vozidel.
    
    Parametry:
    -----------
    video_path : str
        Cesta k vstupnímu video souboru
    model_path : str
        Cesta k YOLOv8 modelu (např. 'yolov8m.pt')
    output_csv : str
        Cesta pro výstupní CSV soubor
    """
    
    # Příprava zařízení (GPU nebo CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Používá se zařízení: {device}")
    
    # Načtení YOLO modelu
    model = YOLO(model_path)
    model.to(device)

    # Konverze zabudovaných dat na Shapely polygony
    roi_polygon = Polygon(ROI_POLYGON_COORDS)
    exit_lines = {key: Polygon(value) for key, value in EXIT_LINES_COORDS.items()}

    # Generování unikátního názvu CSV souboru (aby se nepřepsaly předchozí)
    attempt = 1
    base_output_csv = output_csv
    while os.path.exists(output_csv):
        output_csv = base_output_csv.replace('.csv', f'_attempt{attempt}.csv')
        attempt += 1

    # Otevření videa
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Chyba: Nelze otevřít video soubor {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Výchozí záloha
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Datové struktury pro sledování
    track_state = {}  # track_id -> stav vozidla
    crossing_counts = defaultdict(lambda: defaultdict(int))  # minuta -> exit_id -> počet
    inactive_tracks = {}  # track_id -> počet po sobě jdoucích snímků bez detekce

    TRACK_TIMEOUT = 30  # Smazat vozidlo pokud je 30 snímků neviditelné
    MIN_TRACK_LENGTH = 5  # Minimálně 5 snímků pro platnou detekci
    SHOW_VIDEO = False  # Vypnuto pro batch zpracování (zrychluje běh)

    frame_idx = 0
    print(f"Zpracování {total_frames} snímků na {fps} snímků za sekundu...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            minute = frame_idx // (int(fps) * 60)

            # Detekce vozidel s sledováním (class 2 = auto, class 7 = truck)
            # conf=0.30 - práh spolehlivosti (30% jistota že je vozidlo)
            results = model.track(frame, persist=True, classes=[2, 7], conf=0.30)

            detected_tracks = set()

            # Kreslení výjezdů na snímek (volitelné - pro debugging)
            for exit_id, polygon in exit_lines.items():
                coords = list(polygon.exterior.coords)
                pts = np.array(coords, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                
                exit_count = crossing_counts[minute].get(exit_id, 0)
                centroid_x = int(np.mean([c[0] for c in coords]))
                centroid_y = int(np.mean([c[1] for c in coords]))
                cv2.putText(frame, f"{exit_id}: {exit_count}", (centroid_x - 30, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Zpracování detekčních - pro každé vozidlo
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id)
                    detected_tracks.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Střední bod vozidla (spodek pro stabilitu)
                    center_x = (x1 + x2) // 2
                    bottom_y = y2
                    detection_point = Point(center_x, bottom_y)

                    # Kontrola: je vozidlo v ROI?
                    if roi_polygon.contains(detection_point):

                        # Inicializace stavu vozidla pokud je první detekce
                        if track_id not in track_state:
                            track_state[track_id] = {
                                'detected_in_roi': True,
                                'exits_state': {exit_id: {'inside': False, 'counted': False} 
                                               for exit_id in exit_lines.keys()},
                                'frame_count': 1,
                                'last_position': (center_x, bottom_y)
                            }
                        else:
                            track_state[track_id]['frame_count'] += 1
                            track_state[track_id]['last_position'] = (center_x, bottom_y)

                        # Odebrat z neaktivních
                        if track_id in inactive_tracks:
                            del inactive_tracks[track_id]

                        # Kreslení vozidla (debugging)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Kontrola přechodů přes výjezdní čáry
                        for exit_id, polygon in exit_lines.items():
                            current_inside = polygon.contains(detection_point)
                            previous_inside = track_state[track_id]['exits_state'][exit_id]['inside']
                            counted = track_state[track_id]['exits_state'][exit_id]['counted']

                            # Aktualizace stavu
                            track_state[track_id]['exits_state'][exit_id]['inside'] = current_inside

                            # Počítej při přechodu: byl uvnitř, nyní venku, ještě nepočítáno
                            if previous_inside and not current_inside and not counted:
                                # Validace kvality sledování
                                if track_state[track_id]['frame_count'] >= MIN_TRACK_LENGTH:
                                    track_state[track_id]['exits_state'][exit_id]['counted'] = True
                                    crossing_counts[minute][exit_id] += 1
                                    print(f"Frame {frame_idx}: Track {track_id} vyjel {exit_id} (minuta {minute})")

            # Správa neaktivních vozidel
            for track_id in list(track_state.keys()):
                if track_id not in detected_tracks:
                    if track_id not in inactive_tracks:
                        inactive_tracks[track_id] = 1
                    else:
                        inactive_tracks[track_id] += 1

                    # Smazat po dlouhé neaktivitě
                    if inactive_tracks[track_id] > TRACK_TIMEOUT:
                        del track_state[track_id]
                        del inactive_tracks[track_id]

            # Výstisk průběhu
            if frame_idx % 30 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  Pruběh: {frame_idx}/{total_frames} snímků ({pct:.1f}%)")

            # Zobrazení videa (volitelné)
            if SHOW_VIDEO:
                y_offset = 30
                cv2.putText(frame, f"Min: {minute}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, y_offset + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow('Car Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nZpracování přerušeno uživatelem.")

    # Čistka
    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()

    # ========================================================================
    # ZÁPIS VÝSLEDKŮ DO CSV
    # ========================================================================
    
    if crossing_counts or exit_lines:
        # Všechny výjezdy (včetně těch bez vozidel)
        all_exits = sorted(list(exit_lines.keys()))

        # Počet kompletních minut
        total_minutes = int(total_frames / (int(fps) * 60))

        # Výpočet součtů
        exit_totals = {exit_id: 0 for exit_id in all_exits}
        for minute in range(total_minutes):
            for exit_id in all_exits:
                exit_totals[exit_id] += crossing_counts[minute].get(exit_id, 0)

        # Zápis CSV
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['minute'] + all_exits
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Řádky - minuty
            for minute in range(total_minutes):
                row = {'minute': minute}
                for exit_id in all_exits:
                    row[exit_id] = crossing_counts[minute].get(exit_id, 0)
                writer.writerow(row)

            # Poslední řádek - součty
            total_row = {'minute': 'TOTAL'}
            for exit_id in all_exits:
                total_row[exit_id] = exit_totals[exit_id]
            writer.writerow(total_row)

        print(f"\nAggregované výsledky uloženy do {output_csv}")
        print(f"Celkem vozidel vyjelo:")
        for exit_id in all_exits:
            print(f"  {exit_id}: {exit_totals[exit_id]}")
    else:
        print("Nebyly zjištěny žádné vozidla.")


# ============================================================================
# VSTUPNÍ BOD PROGRAMU
# ============================================================================

if __name__ == "__main__":
    # Cesty k souborům
    video_path = "data/roundabout.avi"
    model_path = "yolov8m.pt"  # Predownload z https://github.com/ultralytics/assets/releases
    output_csv = "output/car_crossings.csv"

    # Spuštění detekce
    detect_cars(video_path, model_path, output_csv)
