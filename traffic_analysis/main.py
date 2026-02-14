import configparser
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import logging
from typing import List, Tuple, Dict, Set
import torch

# --- Funkce pro načtení konfigurace ---
def load_config(config_path: str) -> Dict:
    """Načte konfiguraci z .ini souboru."""
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        if not config.sections():
            raise FileNotFoundError(f"Konfigurační soubor '{config_path}' nebyl nalezen nebo je prázdný.")
        
        settings = {
            'video_path': config.get('general', 'video_path'),
            'model_path': config.get('general', 'model_path'),
            'confidence_threshold': config.getfloat('general', 'confidence_threshold'),
            'visualize': config.getboolean('general', 'visualize'),
            'output_csv': config.get('general', 'output_csv'),
            'counting_lines': {int(k.split('_')[1]): tuple(map(int, v.split(',')))
                               for k, v in config.items('lines')}
        }
        # Převedení na správný formát pro COUNTING_LINES
        settings['counting_lines'] = {
            k: ((v[0], v[1]), (v[2], v[3])) for k, v in settings['counting_lines'].items()
        }
        
        # Načtení ROI, pokud existuje
        if config.has_section('roi') and config.has_option('roi', 'polygon'):
            roi_str = config.get('roi', 'polygon')
            roi_points = np.array([int(p) for p in roi_str.split(',')]).reshape(-1, 2)
            settings['roi_polygon'] = roi_points
        else:
            settings['roi_polygon'] = None

        logging.info("Konfigurace byla úspěšně načtena.")
        return settings
    except (configparser.Error, FileNotFoundError, KeyError) as e:
        logging.error(f"Chyba při načítání konfigurace: {e}")
        raise

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Architektura ---

def load_model(model_path: str) -> YOLO:
    """Načte YOLO model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YOLO(model_path)
        model.to(device)  # Explicitly move the model to the selected device
        logging.info(f"Model '{model_path}' byl úspěšně načten na zařízení: {device}.")
        return model
    except Exception as e:
        logging.error(f"Chyba při načítání modelu: {e}")
        raise

def process_frame(model: YOLO, frame: np.ndarray, confidence_threshold: float, roi_mask: np.ndarray = None) -> List:
    """Zpracuje jeden frame videa a detekuje vozidla."""
    if roi_mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    results = model(frame, classes=[2], conf=confidence_threshold, verbose=True)  # Verbose for debugging
    logging.info(f"Detections: {results[0].boxes.data.cpu().numpy()}")  # Log detections
    return results[0].boxes.data.cpu().numpy()

def update_tracker(detections: List) -> Dict[int, List[Tuple[int, int]]]:
    """
    Aktualizuje tracker a vrací sledované objekty.
    Tato funkce je zjednodušená. Pro reálné použití by zde byla integrace
    s ByteTrack nebo DeepSORT. Pro tento příklad použijeme jednoduchý
    mechanismus založený na detekcích v každém framu.
    """
    # V reálném scénáři by zde byla logika pro přiřazení ID.
    # Pro zjednodušení budeme předpokládat, že každá detekce je nový objekt,
    # což není správné, ale pro demonstraci to stačí.
    # V pokročilejší verzi by se zde volal tracker.update(detections).
    
    tracked_objects = {}
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, _ = det
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        tracked_objects[i] = [centroid] # Simulace historie, i je falešné ID
    return tracked_objects


def check_line_crossing(
    tracked_objects: Dict[int, List[Tuple[int, int]]],
    lines: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    crossed_ids: Set[int]
) -> Tuple[Dict[int, int], Set[int]]:
    """Kontroluje protnutí čar a vrací počet protnutí pro každý výjezd."""
    exit_counts = {exit_id: 0 for exit_id in lines}
    
    for obj_id, history in tracked_objects.items():
        if len(history) < 2:
            continue # Potřebujeme alespoň dva body pro určení směru

        # Zjednodušená kontrola - v reálném případě by se kontroloval skutečný průsečík
        last_point = history[-1]
        
        for exit_id, line in lines.items():
            # Zjednodušená kontrola, zda je bod "blízko" čáry
            # a zda se pohybuje "ven" (velmi zjednodušené)
            # V reálném případě by se použila geometrická kontrola průsečíku.
            
            # Příklad: pokud je auto na výjezdu 1 a pohybuje se nahoru
            if exit_id == 1 and line[0][0] < last_point[0] < line[1][0] and abs(last_point[1] - line[0][1]) < 10:
                 if obj_id not in crossed_ids:
                    exit_counts[exit_id] += 1
                    crossed_ids.add(obj_id)

    return exit_counts, crossed_ids


def aggregate_counts(
    current_counts: Dict[int, Dict[int, int]],
    frame_counts: Dict[int, int],
    frame_idx: int,
    fps: float,
    counting_lines: Dict
) -> Dict[int, Dict[int, int]]:
    """Agreguje počty vozidel po minutách."""
    minute = int(frame_idx / (fps * 60))
    if minute not in current_counts:
        current_counts[minute] = {exit_id: 0 for exit_id in counting_lines}

    for exit_id, count in frame_counts.items():
        current_counts[minute][exit_id] += count
        
    return current_counts

def export_results(counts: Dict[int, Dict[int, int]], output_path: str):
    """Exportuje výsledky do CSV souboru."""
    if not counts:
        logging.warning("Žádná data k exportu.")
        return

    df = pd.DataFrame.from_dict(counts, orient='index')
    df = df.fillna(0).astype(int)
    df.index.name = 'minute'
    df = df.rename(columns={k: f'exit_{k}' for k in df.columns})
    
    try:
        df.to_csv(output_path)
        logging.info(f"Výsledky byly úspěšně exportovány do '{output_path}'.")
    except Exception as e:
        logging.error(f"Chyba při exportu do CSV: {e}")

def visualize_frame(frame: np.ndarray, detections: List, lines: Dict):
    """Vykreslí detekce a čáry na frame."""
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])  # Ensure correct unpacking
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for exit_id, line in lines.items():
        cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
        cv2.putText(frame, f"Exit {exit_id}", (line[0][0], line[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True


def main():
    """Hlavní funkce pro spuštění analýzy."""
    try:
        config = load_config('config.ini')
    except Exception as e:
        logging.error(f"Program bude ukončen kvůli chybě v konfiguraci: {e}")
        return

    model = load_model(config['model_path'])
    
    try:
        cap = cv2.VideoCapture(config['video_path'])
        if not cap.isOpened():
            raise IOError(f"Nelze otevřít video soubor: {config['video_path']}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logging.warning("Nelze zjistit FPS, používá se výchozí hodnota 25.")
            fps = 25

        logging.info(f"Video načteno: {config['video_path']} (FPS: {fps:.2f})")

    except Exception as e:
        logging.error(e)
        return

    frame_idx = 0
    total_counts = {}
    crossed_vehicle_ids = set()

    # Vytvoření masky pro ROI, pokud je definována
    roi_mask = None
    if config.get('roi_polygon') is not None:
        ret, frame = cap.read()
        if ret:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [config['roi_polygon']], 255)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Vrátíme se na začátek videa
        else:
            logging.warning("Nepodařilo se vytvořit ROI masku.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = process_frame(model, frame, config['confidence_threshold'], roi_mask)
        
        # Zde by se v reálné aplikaci použil skutečný tracker
        tracked_objects = update_tracker(detections)
        
        frame_exit_counts, crossed_vehicle_ids = check_line_crossing(
            tracked_objects, config['counting_lines'], crossed_vehicle_ids
        )
        
        total_counts = aggregate_counts(total_counts, frame_exit_counts, frame_idx, fps, config['counting_lines'])

        if config['visualize']:
            if not visualize_frame(frame, detections, config['counting_lines']):
                break
        
        frame_idx += 1

    cap.release()
    if config['visualize']:
        cv2.destroyAllWindows()

    export_results(total_counts, config['output_csv'])
    logging.info("Analýza dokončena.")


if __name__ == "__main__":
    main()
