# Guide de l'Architecture: Deep Symbolic Regression (DSR)

Bienvenue dans la documentation architecturale du projet **Deep Symbolic Regression**. 
Ce document est un guide complet écrit en français pour comprendre comment le réseau de neurones récurrent parvient à littéralement "redécouvrir" les lois de la physique analytique depuis zéro, et comment le code bat en rythme derrière la magie de l'IA.

---

## 1. La Pipeline Globale (Comment ça marche ?)

Si tu lances l'entraînement via `uv run python -m src.dsr.training.train`, voici exactement ce qu'il se passe de A à Z :

1. **Chargement de la matrice (Données)** : Le système interroge la base de données PMLB (`datasets.py`) pour télécharger une énorme équation de physique (ex: la loi de la gravité de Feynman). On récupère un jeu de données de points ($X$, $y$).
2. **Initialisation de l'Agent IA** : Un réseau de neurones complexe à 2 couches (le `SymbolicPolicy` LSTM) s'allume au sein du `Trainer`. Ce réseau est totalement ignorant des lois de la physique.
3. **Génération Massive (Batched GPU)** : C'est la force de frappe. Le `rollout.py` lance la prédiction du réseau de neurones sur ta carte graphique (CUDA) : le LSTM choisit des mots dans un alphabet mathématique (`+`, `*`, `sin`, `x0`, `0.5`, `pi`, `const`) et construit **simultanément 256 trajectoires et arbres mathématiques différents** sans la moindre boucle Python.
4. **Calcul d'Erreur (Évaluation et BFSG)** : Chaque suite syntaxique (`[+, x0, const]`) est passée dans l'évaluateur (`evaluator.py`). Si la formule contient le mot magique `const`, l'algorithme "SciPy BFGS" met l'IA en pause quelques millisecondes, et teste frénétiquement plein de nombres décimaux (ex: $2.45$, $9.81$) pour trouver la force optimale. On compare les résultats aux vrais points de physique via l'erreur "NMSE".
5. **Punition ou Récompense (Apprentissage par renforcement)** : On ne récompense pas la médiocrité. Le `risk_seeking_optimizer.py` isole uniquement le **Top 5%** des formules les plus fantastiques et parfaites des 256. L'erreur rétropropage les gradients des probabilités à travers l'IA uniquement pour ces formules ("Risk-Seeking").
6. **Teacher Forcing** : Pour éviter que le réseau de neurones n'oublie son Eureka, le `rollout.py` réinjecte en permanence dans les poids du modèle les 20 meilleures équations de toute l'histoire de l'entraînement (`TopKMemory`), afin de guider doucement les prochaines trajectoires vers cette architecture syntaxique.

---

## 2. Explication Détaillée des Fichiers

L'architecture est découpée dans l'approche Modèle-Vue-Contrôleur (MVC) modernisée pour l'Intelligence Artificielle.

### 📚 Les Outils d'Exécution ("Les Boutons Rouges")
Situés dans `src/dsr/training/` et `results/`.
- **`src/dsr/training/train.py`** : C'est le point de lancement global de ton projet. Il paramètre ton Intelligence Artificielle, télécharge la série d'équations, et gère la sauvegarde en temps réel dans un fichier Excel/CSV vers le dossier `results/`.
- **`src/dsr/training/run_optuna.py`** : Le fignoleur de "Hyperparameters". Tu lui donnes ton réseau de neurones, et il va tenter 50 mutations différentes (modification du *Learning Rate*, de *l'Entropie*) sur tous les cœurs de ton processeur pour trouver la configuration mathématique absolue qui maximise le QI de ton agent.
- **`results/visualize.py`** : L'artiste de l'équipe. Il lit le dernier fichier Excel des résultats, identifie la formule mathématique valide la plus colossale inventée par le réseau, et utilise la librairie `NetworkX` pour en générer un sublime arbre syntaxique visuel `best_equation.png` sans polluer l'ordinateur.

### 🧠 Le Cerveau (`src/dsr/models/`)
- **`policy.py`** : L'architecture de ton IA. Il contient le `DeepSetsEncoder` (qui permet au réseau de comprendre la position géométrique de tous les points de la courbe cible) et surtout la `SymbolicPolicy` (un LSTM massif de 512 neurones qui gère la dimension "Batch" de ta carte graphique pour prédire des vecteurs de probabilités).

### ⚙️ Le Cœur Algorithmique (`src/dsr/core/`)
- **`config.py`** : Le dictionnaire des lois universelles de ton run. Il détermine les seuils extrêmes (20 tokens au max par formule) et le lexique d'actions (`unary_operators`, `binary_operators`, `constants`). C'est lui qui introduit l'injection physique de  `pi` et `0.5`.
- **`factory.py`** : Il lit simplement `config.py` et accouche de la "Grammaire", dictant au réseau ce qui est formellement autorisé et ce qui est impossible mathématiquement parlant.
- **`expression.py`** : Les mathématiques magiques. Des algorithmes de navigation récursive dans les arbres. Ils convertissent la vase crachée par le réseau (liste `['*', 'x0', '2.0']`) en objet `Node` traversable mathématiquement (Infix: `(x0 * 2.0)`).
- **`evaluator.py`** : La cour d'assises. Transforme en objet appelable les propositions de l'Agent. Il évalue la fonction face aux donnés de la nature, mais surtout, c'est lui qui implémente **la méta-optimisation BFGS** (Trouver la valeur du `const` cachée).
- **`env.py`** : Le "cadre" formel de Gym-Environment qui maintient l'état et prépare des squelettes vides à passer à l'Agent.

### 🧪 Les Algorithmes d'Apprentissage (`src/dsr/training/`)
- **`trainer.py`** : Le grand Chef d'Orchestre ! Il attache le cerceau (l'IA), demande aux ouvriers de générer et d'évaluer, remplit la mémoire des vainqueurs, calcule la moyenne finale de l'entropie et pousse la mise à jour des poids de l'IA (le `loss.backward()`).
- **`rollout.py`** : Un chef-d'œuvre matriciel. Il remplace de vulgaires boucles `for/while` en Python par des opérations tensorielles pure PyTorch (ex: `.unsqueeze(0)`, des masques booléens). C'est lui qui calcule le flux de probabilités de 256 équations à la fois sur ton processeur graphique, générant ce gigantesque x90 de vitesse d'extrapolation. Gère aussi le système de *Teacher Forcing* de la mémoire Top-K.
- **`risk_seeking_optimizer.py`** : (L'élite). C'est le module "State-Of-The-Art" du reinforcement learning qui ne calcule le Gradient global que pour le sous-ensemble exceptionnel `~ 5%` des 256 formules pures. L'agent est repoussé dans ses retranchements au fil des punitions pour découvrir.
- **`policy_optimizer.py` / `ppo_optimizer.py`** : Héritages open-source conservés (REINFORCE traditionnel, Clipping PPO Actor-Critic) en cas de tests expérimentaux plus doux.

### 📊 L'Intelligence des Données (`src/dsr/data/` & `src/dsr/analysis/`)
- **`data/datasets.py`** : Le scrapper intelligent. Il ouvre le protocole HTTP vers les serveurs universitaires afin d'acquérir les matrices PMLB des 100 Équations du physicien Feynman en direct pour nourrir ton réseau. Il a été débarrassé de tout code expérimental ("Nguyen", "Toys").
- **`analysis/memory.py`** : Le cimetière des Dieux. Un système très simple d'une grande priorité (PriorityQueue / HeapQ) qui garde uniquement en mémoire RAM les 20 plus magnifiques équations crées un jour par ton IA.
- **`analysis/visualizer.py`** : Le framework de projection dans Matplotlib des matrices mathématiques, utilisant l'AST et les dictionnaires de couleurs pour sublimer les feuilles et les branches (Variables mathématiques de l'Agent).

---
*Ce projet suit une conception logicielle immaculée, optimisée pour le "Memory Footprint" du CPU et de la VRAM, avec une scalabilité lui permettant de rivaliser avec les bibliothèques GitHub mondiales.*
