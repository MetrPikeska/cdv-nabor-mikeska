import cv2
import json

# Inicializace globálních proměnných
exit_lines = {}
current_line = []
line_count = 0

# Počet linií od uživatele
num_lines = int(input("Zadejte počet exit linií (např. 4 pro kruhový objezd): "))

def click_event(event, x, y, flags, params):
    """Funkce pro zpracování kliknutí myší."""
    global current_line, exit_lines, line_count
    if event == cv2.EVENT_LBUTTONDOWN:
        current_line.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        if len(current_line) == 2:
            line_name = f"exit_{line_count + 1}"
            exit_lines[line_name] = current_line
            cv2.line(image, current_line[0], current_line[1], (0, 255, 0), 2)
            current_line = []
            line_count += 1
            if line_count >= num_lines:
                print("Všechny linie byly nastaveny. Stiskněte 'q' pro ukončení.")
        cv2.imshow('Set Exit Lines', image)

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

# Zobrazení snímku a nastavení exit lines
cv2.imshow('Set Exit Lines', image)
cv2.setMouseCallback('Set Exit Lines', click_event)

print(f"Klikněte levým tlačítkem pro nastavení dvou bodů každé exit line. Počet linií: {num_lines}")
print("Stiskněte klávesu 'q' pro ukončení.")
cv2.waitKey(0)

# Uložení exit lines do JSON souboru
exit_lines_file = 'output/exit_lines.json'
with open(exit_lines_file, 'w') as f:
    json.dump(exit_lines, f)

print(f"Exit lines byly uloženy do souboru {exit_lines_file}")

# Uvolnění prostředků
cap.release()
cv2.destroyAllWindows()