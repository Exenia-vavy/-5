from ultralytics import YOLO
import os

# 1. Définit le chemin de ton fichier vidéo ici
# Utilise un 'r' devant les guillemets pour éviter les problèmes de caractères spéciaux (Windows)
video_path = r"C:\Users\Exenia\Desktop\videoss\m.mp4"

# 2. Vérification simple si le fichier existe vraiment
if not os.path.exists(video_path):
    print(f"Erreur : Le fichier au chemin {video_path} est introuvable.")
else:
    # 3. Charger le modèle (YOLOv8 nano est parfait pour débuter)
    model = YOLO("yolov8n.pt") 

    # 4. Lancer la détection
    # conf=0.5 signifie qu'on ne garde que les détections sûres à plus de 50%
    results = model.predict(source=video_path, save=True, conf=0.5)

    print(f"Succès ! La vidéo annotée est enregistrée dans le dossier : runs/detect/predict")
