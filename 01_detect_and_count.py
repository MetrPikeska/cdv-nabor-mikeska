import cv2
from ultralytics import YOLO
import torch
import json
from shapely.geometry import Point, Polygon, LineString
import csv
import os
import numpy as np
from collections import defaultdict

print("CUDA available:", torch.cuda.is_available())

def detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv):
    
    # Načtení modelu YOLO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Používá se zařízení: {device}")
    model = YOLO(model_path)
    model.to(device)

    # Načtení polygonu ROI
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    roi_polygon = Polygon(roi_data[0])

    # Načtení výjezdních čar/polygonů
    with open(exit_lines_path, 'r') as f:
        exit_lines_data = json.load(f)
    exit_lines = {key: Polygon(value) for key, value in exit_lines_data.items()}

    # Načtení polygonu vyloučení (pokud existuje)
    exclusion_polygon = None
    exclusion_path = "output/exclusion.json"
    if os.path.exists(exclusion_path):
        with open(exclusion_path, 'r') as f:
            exclusion_data = json.load(f)
        exclusion_polygon = Polygon(exclusion_data[0])

    # Vygenerování unikátního názvu výstupního CSV souboru
    attempt = 1
    base_output_csv = output_csv
    while os.path.exists(output_csv):
        output_csv = base_output_csv.replace('.csv', f'_attempt{attempt}.csv')
        attempt += 1

    # Otevření videa k získání FPS a počtu snímků
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Chyba: Nelze otevřít video soubor {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Výchozí záloha
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Správa stavu pro sledování
    # track_id -> {
    #   'detected_in_roi': bool,
    #   'exits_state': {exit_id: {'inside': bool, 'counted': bool}},
    #   'frame_count': int,
    #   'last_position': (x, y)
    # }
    track_state = {}
    crossing_counts = defaultdict(lambda: defaultdict(int))  # minuta -> exit_id -> počet
    inactive_tracks = {}  # track_id -> neaktivní_snímky
    
    TRACK_TIMEOUT = 30  # snímky
    MIN_TRACK_LENGTH = 5  # minimální snímky, které se mají považovat za platné
    MIN_MOVEMENT = 10  # minimální pohyb pixelů, aby se počítal jako platný
    SHOW_VIDEO = True  # Nastavit na False pro zpracování bez zobrazení, True pro zobrazení videa
    
    frame_idx = 0
    print(f"Zpracování {total_frames} snímků na {fps} snímků za sekundu...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            minute = frame_idx // (int(fps) * 60)
            
            # Získání detekčních se sledováním (třída 2 = auto, třída 7 = nákladní vůz)
            results = model.track(frame, persist=True, classes=[2, 7], conf=0.30)
            
            detected_tracks = set()
            
            # Kreslení polygonů výjezdů na snímek
            for exit_id, polygon in exit_lines.items():
                coords = list(polygon.exterior.coords)
                pts = np.array(coords, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                
                # Přidání čítače pro tento výjezd
                exit_count = crossing_counts[minute].get(exit_id, 0)
                # Nalezení těžiště polygonu pro umístění textu
                centroid_x = int(np.mean([c[0] for c in coords]))
                centroid_y = int(np.mean([c[1] for c in coords]))
                cv2.putText(frame, f"{exit_id}: {exit_count}", (centroid_x - 30, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Zpracování detekčních
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    
                    track_id = int(box.id)
                    detected_tracks.add(track_id)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    
                    # Použití spodního-středního bodu pro stabilnější detekci přejezdu
                    center_x = (x1 + x2) // 2
                    bottom_y = y2  # Spodek ohraničujícího rámečku
                    detection_point = Point(center_x, bottom_y)
                    
                    # Kontrola, zda je v ROI a ne v zóně vyloučení
                    if roi_polygon.contains(detection_point) and (exclusion_polygon is None or not exclusion_polygon.contains(detection_point)):
                        
                        # Inicializace stavu sledování, pokud je potřeba
                        if track_id not in track_state:
                            track_state[track_id] = {
                                'detected_in_roi': True,
                                'exits_state': {exit_id: {'inside': False, 'counted': False} for exit_id in exit_lines.keys()},
                                'frame_count': 1,
                                'last_position': (center_x, bottom_y)
                            }
                        else:
                            track_state[track_id]['frame_count'] += 1
                            track_state[track_id]['last_position'] = (center_x, bottom_y)
                        
                        # Odebrání ze sledování neaktivního
                        if track_id in inactive_tracks:
                            del inactive_tracks[track_id]
                        
                        # Kreslení ohraničujícího rámečku
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Kontrola přechodů stavu polygonu výjezdu (počítání založené na stavu)
                        for exit_id, polygon in exit_lines.items():
                            current_inside = polygon.contains(detection_point)
                            previous_inside = track_state[track_id]['exits_state'][exit_id]['inside']
                            counted = track_state[track_id]['exits_state'][exit_id]['counted']
                            
                            # Aktualizace aktuálního stavu
                            track_state[track_id]['exits_state'][exit_id]['inside'] = current_inside
                            
                            # Počítej pouze při přechodu: byl uvnitř, nyní venku a ještě nebyl počítán
                            if previous_inside and not current_inside and not counted:
                                # Ověrení kvality sledování před počítáním
                                if track_state[track_id]['frame_count'] >= MIN_TRACK_LENGTH:
                                    # Označit jako spočítaný, aby se zabránilo duplicitnímu počítání
                                    track_state[track_id]['exits_state'][exit_id]['counted'] = True
                                    crossing_counts[minute][exit_id] += 1
                                    print(f"Snímek {frame_idx}: Sledování {track_id} opustilo {exit_id} (minuta {minute})")
            
            # Sledování neaktivních vozidel
            for track_id in list(track_state.keys()):
                if track_id not in detected_tracks:
                    if track_id not in inactive_tracks:
                        inactive_tracks[track_id] = 1
                    else:
                        inactive_tracks[track_id] += 1
                    
                    # Odstranit, pokud je moc dlouho neaktivní
                    if inactive_tracks[track_id] > TRACK_TIMEOUT:
                        del track_state[track_id]
                        del inactive_tracks[track_id]
            
            # Zobrazení průběhu
            if frame_idx % 30 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  Průběh: {frame_idx}/{total_frames} snímků ({pct:.1f}%)")
            
            # Přidání globálního zobrazení čítače v levém horním rohu
            y_offset = 30
            cv2.putText(frame, f"Minuta: {minute}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Snímek: {frame_idx}/{total_frames}", (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Sledovaná: {len(track_state)}", (10, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
            
            # Přidání statistik v pravém dolním rohu (dva stoly vedle sebe)
            frame_height, frame_width = frame.shape[:2]
            
            # Výpočet kumulativních součtů od začátku
            cumulative_totals = {exit_id: 0 for exit_id in exit_lines.keys()}
            for m in range(minute + 1):
                for exit_id in exit_lines.keys():
                    cumulative_totals[exit_id] += crossing_counts[m].get(exit_id, 0)
            
            # Rozložení: dva sloupce vedle sebe
            box_height = 60 + len(exit_lines) * 22
            col_width = 150
            total_box_width = col_width * 2 + 20  # Dva sloupce + mezera
            
            box_x = frame_width - total_box_width - 10
            box_y = frame_height - box_height - 10
            
            # Kreslení bílého obdélníku na pozadí pro oba sloupce
            cv2.rectangle(frame, (box_x, box_y), (frame_width - 10, frame_height - 10), (255, 255, 255), -1)
            cv2.rectangle(frame, (box_x, box_y), (frame_width - 10, frame_height - 10), (0, 0, 0), 2)
            
            # LEVÝ SLOUPEC: Výjezdy tuto minutu
            text_y = box_y + 25
            cv2.putText(frame, "Tuto minutu:", (box_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            text_y += 22
            
            for exit_id in sorted(exit_lines.keys()):
                exit_count = crossing_counts[minute].get(exit_id, 0)
                cv2.putText(frame, f"{exit_id}: {exit_count}", (box_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
                text_y += 20
            
            # PRAVÝ SLOUPEC: Celkem od začátku
            text_y = box_y + 25
            cv2.putText(frame, "Celkem:", (box_x + col_width + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            text_y += 22
            
            for exit_id in sorted(exit_lines.keys()):
                total_count = cumulative_totals[exit_id]
                cv2.putText(frame, f"{exit_id}: {total_count}", (box_x + col_width + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 1)
                text_y += 20
            
            # Zobrazení snímku (volitelné)
            if SHOW_VIDEO:
                cv2.imshow('Detekce vozidel', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nZpracování přerušeno uživatelem.")
    
    # Uvolnění prostředků
    cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    
    # Zápis agregovaných výsledků do CSV
    if crossing_counts or exit_lines:
        # Všechny výjezdy z konfigurace (včetně nul)
        all_exits = sorted(list(exit_lines.keys()))
        
        # Správný počet minut (pouze úplné minuty)
        total_minutes = int(total_frames / (int(fps) * 60))
        
        # Výpočet součtů pro každý výjezd (pouze do platných minut)
        exit_totals = {exit_id: 0 for exit_id in all_exits}
        for minute in range(total_minutes):
            for exit_id in all_exits:
                exit_totals[exit_id] += crossing_counts[minute].get(exit_id, 0)
        
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['minute'] + all_exits
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Zápis dat pro každou platnou minutu (0 až total_minutes-1)
            for minute in range(total_minutes):
                row = {'minute': minute}
                for exit_id in all_exits:
                    row[exit_id] = crossing_counts[minute].get(exit_id, 0)
                writer.writerow(row)
            
            # Zápis řádku součtů
            total_row = {'minute': 'TOTAL'}
            for exit_id in all_exits:
                total_row[exit_id] = exit_totals[exit_id]
            writer.writerow(total_row)
        
        print(f"\nAgregované výsledky uloženy do {output_csv}")
        print(f"Celkem výjezdů: {exit_totals}")
    else:
        print("Nebyly zjištěny žádné přejezdy.")



if __name__ == "__main__":
    video_path = "data/roundabout.avi"
    model_path = "yolov8m.pt"
    roi_path = "output/roi.json"
    exit_lines_path = "output/exit_lines.json"
    output_csv = "output/car_crossings.csv"

    detect_cars(video_path, model_path, roi_path, exit_lines_path, output_csv)