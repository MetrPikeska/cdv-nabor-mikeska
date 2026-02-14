import cv2
import json

# Inicializace globálních proměnných
exclusion_polygons = []
current_polygon = []

def click_event(event, x, y, flags, params):
    """Funkce pro zpracování kliknutí myší."""
    global current_polygon, exclusion_polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        if len(current_polygon) > 1:
            cv2.line(image, current_polygon[-2], current_polygon[-1], (255, 0, 0), 2)
        cv2.imshow('Set Exclusion ROI', image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_polygon:
            # Uzavření polygonu
            cv2.line(image, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
            exclusion_polygons.append(current_polygon)
            current_polygon = []
            cv2.imshow('Set Exclusion ROI', image)

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

# Zobrazení snímku a nastavení ROI
cv2.imshow('Set Exclusion ROI', image)
cv2.setMouseCallback('Set Exclusion ROI', click_event)

print("Klikněte levým tlačítkem pro přidání bodů do polygonu.")
print("Klikněte pravým tlačítkem pro uzavření aktuálního polygonu.")
print("Stiskněte klávesu 'q' pro ukončení.")
cv2.waitKey(0)

# Uložení ROI do JSON souboru
exclusion_file = 'output/exclusion.json'
with open(exclusion_file, 'w') as f:
    json.dump(exclusion_polygons, f)

print(f"Exclusion ROI polygony byly uloženy do souboru {exclusion_file}")

# Uvolnění prostředků
cap.release()
cv2.destroyAllWindows()