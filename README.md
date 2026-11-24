# TP – Modèles génératifs avec PyTorch (MNIST)

Création, entraînement, évaluation et exploration de modèles sur le dataset MNIST :

1. **Modèle discriminatif** : classifieur MLP (≈ 96 % d'accuracy sur le test, mesurée avec `torcheval`)
2. **Autoencodeur simple** : espace latent 2D, reconstruction, visualisation, génération d'un chiffre moyen
3. **VAE** : perte ELBO (BCE + KL), reconstruction, espace latent, génération par échantillonnage, exploration d'une grille latente

Notebook : [`Aicha_modeles_generatifs.ipynb`](Aicha_modeles_generatifs.ipynb) (sorties incluses).

## Installation (UV)

```bash
uv sync                      # crée .venv et installe les dépendances
source .venv/bin/activate    # Windows : .venv\Scripts\activate
```

Ouvrir ensuite le notebook dans VS Code et choisir le kernel `.venv`.
MNIST est téléchargé automatiquement dans `./data` (ignoré par git).

## Reproduire le projet depuis zéro

```bash
uv init --python=3.11
uv venv
uv add ipykernel torch torchvision torcheval matplotlib
```

## Résultats

| Modèle | Métrique (test) |
|---|---|
| Classifieur | Accuracy ≈ 96,29 % |
| Autoencodeur | MSE ≈ 0,0451 |
| VAE | Loss ≈ 144,0 / image (Rec 137,7 + KL 6,4) |
