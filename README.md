#TDVineQuality
Choix de la RandomForest : 
Nous avons choisi d'utliser la RandomForest car c'est ce qui était le plus simple à mettre en place et le plus adapter à notre problématique.


Pour le vin parfait : 
Nous avons sélectionné les trois colonnes ayant le plus d'impact sur la qualité du vin, à savoir "citric acid", "sulphates" et " alcohol". Cela à été calculé grâce à la corrélation des variables.
Ensuite après une étude des moyennes pour chaque qualité du vin, nous avons remarqué que plus haut sont ses valeurs plus haut est la qualité du vin.
Nous avons donc pris les valeurs les plus hautes présentes dans le jeu de donnés pour ces trois colonnes.
Concernant les autres colonnes, étant donné qu'elles avaient peu d'impact sur la qualité nous avons juste pris leurs moyennes.
Ayant nous avons créer un parfait qui pourrait être considéré comme parfait.


Installation de l'applicatif :
source .venv/bin/activate
uvicorn main:app

