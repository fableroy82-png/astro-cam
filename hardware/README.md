# Mécanique et impression 3D

Sources du dossier de projet de l'auteur, ajoutées le 17 octobre 2026. Noms et organisation relative conservés, notamment le sous-dossier `Projet suiveur astro`.

## Formats

- `.par` : pièces Solid Edge éditables.
- `.asm` : assemblages Solid Edge.
- `.cfg` : configurations associées aux assemblages, conservées comme dépendances CAO utiles.
- `.3mf` : projets d'impression 3D avec géométrie et réglages du trancheur.

Ces fichiers binaires sont des sources de conception utiles, pas des exécutables ni des fichiers temporaires. Les journaux `.log` d'export sont exclus.

## Contenu

À la racine : modèle Arduino Q, assemblage et pièces de boîtier, supports laser haut/bas, obturateur, `Pièce1` et projet `test astro cam`. Les noms génériques et les variantes sont conservés car leurs rôles exacts ou leur interchangeabilité ne sont pas confirmés.

Dans [Projet suiveur astro](Projet%20suiveur%20astro/) : assemblage `Asm2`, écrou et supports de caméra, dont une variante V2. Les variantes sont des modèles distincts ; aucun modèle n'a été déclaré obsolète sans preuve.

Les dix conteneurs 3MF ont été inspectés : leur géométrie et leurs réglages sont présents. Les fichiers CAO natifs ont été copiés sans modification. L'ouverture dans Solid Edge, la résolution de toutes les références d'assemblage et une nouvelle impression n'ont pas été testées. Si Solid Edge demande de retrouver une pièce, sélectionner la pièce correspondante dans ce dossier.

Avant impression, contrôler unités, dimensions, matériau, orientation, supports et profil d'imprimante. Les réglages intégrés sont ceux du projet d'origine.
