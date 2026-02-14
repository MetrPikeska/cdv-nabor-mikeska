import cv2
import json
import numpy as np

# Inicializace globálních proměnných
exit_polygons = {}
current_polygon = []
polygon_id = 1

def click_event(event, x, y, flags, params):
    """Funkce pro zpracování kliknutí myší."""
    global current_polygon, exit_polygons, polygon_id
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        if len(current_polygon) > 1:
            # Nakreslení čáry mezi posledními dvěma body
            cv2.line(image, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
        cv2.imshow('Set Exit Polygons', image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            # Uzavření polygonu - nakreslení čáry od posledního bodu k prvnímu
            cv2.line(image, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
            # Vyplnění polygonu poloprůhlednou barvou
            pts = np.array([current_polygon], dtype=np.int32)
            cv2.polylines(image, pts, True, (0, 255, 100), 2)
            cv2.fillPoly(image, pts, (100, 200, 100))
            
            # Uložení aktuálního polygonu
            exit_polygons[f"exit_{polygon_id}"] = current_polygon.copy()
            current_polygon = []
            polygon_id += 1
            print(f"Polygon exit_{polygon_id - 1} byl uložen.")
            cv2.imshow('Set Exit Polygons', image)
        else:
            print("Musíte zadat alespoň 3 body pro polygon.")

# Načtení videa
video_path = 'data/roundabout.avi'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Chyba: Nelze otevřít video {video_path}")
    exit()

# Načtení prvního snímku
ret, image = cap.read()
if not ret:
    print("Chyba: Nelze načíst první snímek videa.")
    cap.release()
    exit()

# Zobrazení snímku a nastavení výstupních polygonů
cv2.imshow('Set Exit Polygons', image)
cv2.setMouseCallback('Set Exit Polygons', click_event)

print("Klikněte levým tlačítkem pro přidání bodů do polygonu.")
print("Klikněte pravým tlačítkem pro uzavření a uložení polygonu.")
print("Potřebujete alespoň 3 body na polygon.")
print("Stiskněte klávesu 'q' pro ukončení.")
cv2.waitKey(0)

# Uložení výstupních polygonů do JSON souboru
exit_polygons_file = 'output/exit_lines.json'
with open(exit_polygons_file, 'w') as f:
    json.dump(exit_polygons, f)

print(f"\nVýstupní polygony byly uloženy do souboru {exit_polygons_file}")
print(f"Celkový počet polygonů: {len(exit_polygons)}")

# Uvolnění prostředků
cap.release()
cv2.destroyAllWindows()