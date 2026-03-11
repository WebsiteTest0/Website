# MNIST Flutter Web Demo

Diese Version laeuft komplett statisch im Browser und kann auf GitHub Pages gehostet werden.

## Einmalig Modell erzeugen

1. Python-Abhaengigkeiten installieren:
   `pip install -r requirements.txt`
2. Browser-Modell erzeugen:
   `python tool/train_mnist.py`

Dabei wird `assets/models/mnist_dense.json` erstellt.

## Lokal starten

1. Flutter-Abhaengigkeiten laden:
   `flutter pub get`
2. Web-App starten:
   `flutter run -d chrome`

## Fuer GitHub Pages bauen

1. Stelle sicher, dass dein GitHub-Repository bereits existiert, zum Beispiel `mnist-demo`.
2. Baue die Web-App mit deinem Repo-Namen als Base-Href:
   `flutter build web --release --base-href /REPO-NAME/`
3. Wenn dein Repo also `mnist-demo` heisst, ist der Befehl:
   `flutter build web --release --base-href /mnist-demo/`
4. Danach liegt die fertige Website in `build/web/`.
5. Wichtig: Die Datei `.nojekyll` aus `build/web/` muss mit hochgeladen werden.

## Einfache GitHub-Pages-Variante mit `docs/`

1. App bauen:
   `flutter build web --release --base-href /REPO-NAME/`
2. Inhalt von `build/web/` nach `docs/` kopieren.
3. Alles nach GitHub pushen.
4. In GitHub unter `Settings > Pages` als Source `Deploy from a branch` waehlen.
5. Branch `main` und Ordner `/docs` auswaehlen.

## Wichtige Dateien

- `tool/train_mnist.py`: trainiert das MNIST-Modell und exportiert es als JSON fuer Flutter Web
- `assets/models/mnist_dense.json`: Browser-Modell fuer die statische Seite
- `lib/main.dart`: Zeichenflaeche, Slideshow und Inferenz direkt in Flutter

## Hinweis

Nach jeder Modell-Aenderung `python tool/train_mnist.py` erneut ausfuehren und danach die Web-App neu bauen.
