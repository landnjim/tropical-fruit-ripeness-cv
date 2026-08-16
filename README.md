# Estimation de la Maturité des Fruits Tropicaux par Vision par Ordinateur

Projet de classification d'images pour estimer le stade de maturité (mûr / pas mûr) de trois fruits tropicaux — mangue, banane, ananas — à l'aide d'un modèle de deep learning par transfert d'apprentissage.

Travail réalisé dans le cadre d'un TP de Master 2, Université de Kinshasa (UNIKIN), Faculté des Sciences et Technologies.

## Contexte

L'évaluation de la maturité des fruits est traditionnellement effectuée par inspection visuelle humaine. Ce projet explore l'utilisation de la vision par ordinateur pour automatiser cette tâche, avec des applications potentielles dans le tri post-récolte et le contrôle qualité agroalimentaire.

## Jeu de données

- **Mangue et banane** : [Fruit-Ripeness-Detection-Dataset](https://huggingface.co/datasets/darthraider/fruit-ripeness-detection-dataset) (Hugging Face), issu de Mendeley Data — ~5000 images RGB, 2 classes par fruit (mûr / pas mûr)
- **Ananas** : [Pineapple Dataset](https://www.kaggle.com/datasets/adhilpk/pineapple) (Kaggle) — images annotées au format YOLO, reclassées en mûr / pas mûr

6 classes au total : `mango_unripe`, `mango_ripe`, `banana_unripe`, `banana_ripe`, `pineapple_unripe`, `pineapple_ripe`.

## Méthode

- **Architecture** : transfert d'apprentissage avec MobileNetV2 pré-entraîné (ImageNet), couches supérieures gelées, tête de classification personnalisée (GlobalAveragePooling + Dense + Dropout)
- **Prétraitement** : redimensionnement 224×224, normalisation MobileNetV2, augmentation de données (flip horizontal, luminosité aléatoire)
- **Entraînement** : Adam, early stopping sur la validation loss, 15 epochs max

## Résultats

- **Accuracy sur le jeu de test : 98%**
- Mangue et banane classées quasi parfaitement (F1-score ≈ 1.00)
- Ananas légèrement plus faible (F1-score 0.91–0.95), en raison d'un léger déséquilibre de classes dans le dataset source

Voir `/report/training_curves.png` et `/report/confusion_matrix.png` pour le détail.

## Structure du dépôt

## Installation et utilisation

### Application Streamlit

```bash
cd app
pip install streamlit tensorflow pillow numpy
streamlit run app.py
```

L'application permet d'uploader une image de mangue, banane ou ananas et retourne le stade de maturité prédit avec le niveau de confiance.

### Notebook

Le notebook `notebooks/fruit_ripeness_model.ipynb` reproduit l'intégralité du pipeline : téléchargement des données, prétraitement, entraînement, évaluation. Conçu pour être exécuté sur Google Colab avec accélération GPU.

## Limites

- Classification binaire simplifiée (mûr/pas mûr) plutôt que des stades de maturité fins, par contrainte de temps
- Le modèle généralise moins bien sur des images en contexte naturel (fruit sur l'arbre, arrière-plan complexe) que sur des images isolées proches de la distribution d'entraînement
- Léger déséquilibre de classes pour l'ananas (509 vs 940 images)

## Auteur

Kasongo Njiminy Landers — Master 2, UNIKIN
