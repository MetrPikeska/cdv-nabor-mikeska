import cv2
import numpy as np
import configparser
import logging

# --- Konfigurace ---
CONFIG_PATH = 'config.ini'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Globální proměnné pro ukládání bodů
points = []
drawing = False
temp_line_start = None

def mouse_callback(event, x, y, flags, param):
    """Zpracovává události myši pro kreslení."""
    global points, drawing, temp_line_start, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing: # Začátek kreslení polygonu
            points.append((x, y))
            logging.info(f"Přidán bod polygonu: {(x, y)}")
        else: # Kreslení čar
            if temp_line_start is None:
                temp_line_start = (x, y)
                logging.info(f"Začátek čáry: {(x, y)}")
            else:
                points.append((temp_line_start, (x, y)))
                cv2.line(frame_copy, temp_line_start, (x, y), (255, 255, 0), 2)
                logging.info(f"Přidána čára: {temp_line_start} -> {(x, y)}")
                temp_line_start = None

def draw_instructions(frame, text):
    """Vykreslí instrukce na obrazovku."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    global points, drawing, temp_line_start, frame_copy

    config = configparser.ConfigParser()
    try:
        config.read(CONFIG_PATH)
        video_path = config.get('general', 'video_path')
    except Exception as e:
        logging.error(f"Chyba při čtení konfiguračního souboru '{CONFIG_PATH}': {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Nelze otevřít video: {video_path}")
        return
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logging.error("Nelze načíst první frame videa.")
        return

    original_frame = frame.copy()
    frame_copy = frame.copy()
    window_name = "Nastaveni ROI a Car"

    # 1. Krok: Vykreslení polygonu pro ROI
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    instructions = "1. Naklikejte body pro ROI (polygon). Stisknete 'd' pro dokonceni."
    logging.info("--- Krok 1: Definice ROI polygonu ---")
    logging.info("Naklikejte levým tlačítkem myši body, které ohraničí zájmovou oblast (kruhový objezd).")
    logging.info("Po dokončení stiskněte klávesu 'd'.")

    while True:
        frame_display = frame_copy.copy()
        if len(points) > 1:
            cv2.polylines(frame_display, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        draw_instructions(frame_display, instructions)
        cv2.imshow(window_name, frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            if len(points) < 3:
                logging.warning("ROI polygon musí mít alespoň 3 body.")
                continue
            drawing = True
            roi_points = points.copy()
            points = []
            logging.info(f"ROI polygon dokončen s body: {roi_points}")
            break
        elif key == 27: # ESC
            cv2.destroyAllWindows()
            return

    # 2. Krok: Vykreslení počítacích čar
    frame_copy = original_frame.copy()
    cv2.polylines(frame_copy, [np.array(roi_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    instructions = "2. Nakreslete cary pro vyjezdy (klik-klik). Stisknete 's' pro ulozeni."
    logging.info("\n--- Krok 2: Definice počítacích čar ---")
    logging.info("Nyní nakreslete čáry pro každý výjezd. Každá čára je definována dvěma kliky.")
    logging.info("Po nakreslení všech čar stiskněte klávesu 's' pro uložení do konfigurace.")

    while True:
        frame_display = frame_copy.copy()
        draw_instructions(frame_display, instructions)
        cv2.imshow(window_name, frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not points:
                logging.warning("Nebyly definovány žádné čáry.")
                continue
            
            # Uložení do konfigurace
            config.add_section('roi')
            roi_str = ','.join([f"{p[0]},{p[1]}" for p in roi_points])
            config.set('roi', 'polygon', roi_str)
            
            # Smazání starých čar a přidání nových
            if config.has_section('lines'):
                config.remove_section('lines')
            config.add_section('lines')
            for i, line in enumerate(points):
                line_str = f"{line[0][0]},{line[0][1]},{line[1][0]},{line[1][1]}"
                config.set('lines', f'exit_{i+1}', line_str)
            
            with open(CONFIG_PATH, 'w') as configfile:
                config.write(configfile)
            
            logging.info(f"Konfigurace byla úspěšně uložena do '{CONFIG_PATH}'.")
            break
        elif key == 27: # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
