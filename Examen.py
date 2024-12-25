# -*- coding: utf-8 -*-
"""Copie de Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gs4ISpBLDfcExMrrtu0S-wHOHbgioMif

# **PROJET BIOSTATISTIQUE**

---

# **Table des matières**
"""



"""1.   **Importation des bibliothèques de bases**
2.   **Chargement et Description des Données**
3.   **Analyse Exploratoire**
  1.  **Analyse de forme**
  2.  **Analyse de fond**
      1. **Analyse univarié**
      2. **Analyse Multivariée**


4.   **Modélisation**

## 1. **Importation des bibliothèques de bases**
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

"""# **Chargement et description**"""

#chargement
train=pd.read_excel('drive/MyDrive/Donnnées_Projet_M2SID2023_2024.xlsx')
train.head()

"""# **Analyse exploratoire**

### **Analyse de Forme**
"""

print ("Taille des données d'entraînement:", train.shape)

#type de variables de la base en nombre
train.dtypes.value_counts()

#Analyse des données manquantes (par colonne)
train.isna().sum(axis = 0)

#type de variables de la base
train.dtypes

#nombre de modalités/valeurs uniques de chaque variable
train.nunique()

train.Evolution.describe()

#Liste des variables qualitatives
cat_features = train.select_dtypes(exclude = np.number).columns.tolist()
cat_features

#Liste des variables quantitatives
num_features = train.select_dtypes(include=['float64', 'int64']).columns
num_features

"""## **Analyse de Fond**

### **Analyse univariée**
"""

#distribution de chaque variable
for var in num_features :
    plt.figure(figsize = ( 5 , 3))
    sns.histplot(data = train , x = var , stat = "density" , label = "Histogramme")
    sns.kdeplot(data = train , x = var , label = "Courbe de densité", color = "red")
    plt.xlabel(var)
    plt.ylabel("Densité")
    plt.title(f"Distribution de la variable {var}")
    plt.legend()
    # plt.savefig("hist1.jpg" , dpi = 300)
    plt.show()

#Analyse des outliers (valeurs aberrantes) : boxplot
for var in num_features :
    plt.figure(figsize = (5 , 3))
    sns.boxplot(data = train , x = var)
    plt.show()

#Diagramme en barre
# sns.countplot(train[var]).set_tittle("")
for var in cat_features :
    plt.figure(figsize = (5 , 3))
    train[var].value_counts().plot.bar()
    plt.show()

# définition d'une fonction d'imputation des outliers
def imputeOutliers(df , x ):
    Q1 = df[x].quantile(0.25) #quartile d'ordre 1
    Q3 = df[x].quantile(0.75) #quartile d'ordre 3
    IQR = Q3 - Q1 #écart interquartile
    min = Q1 - 1.5*IQR
    max = Q3 + 1.5*IQR
    df.loc[df[x] < min , x ] = min
    df.loc[df[x] > max , x ] = max
    return df

#copy de la base de données
dfNoOutliers = train.copy()
#imputation des outliers
for var in num_features :
   dfNoOutliers = imputeOutliers(dfNoOutliers , var)

#Analyse des outliers (valeurs aberrantes) : boxplot
for var in num_features :
    plt.figure(figsize = (5 , 3))
    sns.boxplot(data = dfNoOutliers , x = var)
    plt.show()

"""### **Analyse multivariée**"""

from scipy.stats import ttest_ind
num_independant=[]
# Exemple : t-test entre deux groupes (décès et non-décès)
for var in num_features:
    group_1 = dfNoOutliers[dfNoOutliers['Evolution'] == "Deces"][var]
    group_2 = dfNoOutliers[dfNoOutliers['Evolution'] == "Vivant"][var]

    stat, p_value = ttest_ind(group_1, group_2, equal_var=False)
    print(f"{var}: Statistique : {stat}, p-valeur : {p_value}")
    if p_value<0.05:
        num_independant.append(var)
print(f"\n\nles variables pertinentes sont \n{num_independant}\nIls sont au nombres de: {len(num_independant)}")

from scipy.stats import chi2_contingency

cat_independant=[]
for var in cat_features:  # Liste des variables catégoriques
    # Tableau de contingence
    contingency_table = pd.crosstab(dfNoOutliers[var], dfNoOutliers['Evolution'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Variable : {var} | Chi2 : {chi2}, p-valeur : {p}")
    if p<0.05:
        cat_independant.append(var)
print(f"\n\nles variables pertinentes sont \n{cat_independant}\nIls sont au nombres de: {len(cat_independant)}")

"""#### Graphique de correlation pour les variables qualitative"""

sns.pairplot(dfNoOutliers, hue='Evolution', palette='Set2', kind='scatter')
plt.show()

for var in cat_features:
    # Exemple de tableau de contingence
    contingency_table = pd.crosstab(dfNoOutliers['Evolution'], dfNoOutliers[var])

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Heatmap des fréquences entre {var} et Evolution")
    plt.ylabel("Evolution ")
    plt.xlabel(f"{var}")
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

for var in cat_features:
    # Création d'un tableau de contingence pour les pourcentages
    contingency_table = pd.crosstab(dfNoOutliers['Evolution'], dfNoOutliers[var], normalize='index')

    # Graphique en barres empilées
    ax = contingency_table.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')

    # Ajouter des étiquettes
    for bar in ax.patches:
        # Calculer la position de l'étiquette
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_y() + bar.get_height() / 2

        # Texte de l'étiquette (proportion avec 2 décimales)
        label = f"{bar.get_height():.2f}"

        # Ajouter l'étiquette uniquement si la hauteur est non nulle
        if bar.get_height() > 0:
            ax.text(x, y, label, ha='center', va='center', fontsize=8, color='black')

    # Personnalisation du graphique
    plt.title(f"Graphique en barres empilées pour {var}")
    plt.ylabel("Proportion")
    plt.xlabel("Evolution")
    plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

pip install statsmodels

for var in cat_features:
    from statsmodels.graphics.mosaicplot import mosaic
    if var != "Evolution":
      # Création du mosaïque plot
      plt.figure(figsize=(8, 6))
      mosaic(dfNoOutliers, ['Evolution', var], title="Mosaïque Plot")
      plt.ylabel(f"{var}")
      plt.xlabel("Evolution")
      plt.legend(title=var)
      plt.show()

"""#### Graphique de correlation pour les variables quantitative

##### Boxplot
"""

for var in num_features:
    # Exemple de données
    sns.boxplot(x='Evolution', y=var, data=dfNoOutliers)
    plt.title(f"Boxplot : {var} vs Evolution")
    plt.xlabel("Evolution")
    plt.ylabel(f"{var}")
    plt.show()

"""##### Violin Plot"""

for var in num_features:
    # Exemple de données
    sns.violinplot(x='Evolution', y=var, data=dfNoOutliers, palette='muted')
    plt.title(f"Violin Plot : {var} vs Evolution")
    plt.xlabel("Evolution")
    plt.ylabel(f"{var}")
    plt.show()

"""##### GRAPHIQUE EN BARRE"""

for var in num_features:
    # Création du barplot pour chaque variable quantitative
    sns.barplot(x='Evolution', y=var, data=dfNoOutliers, errorbar=None)  # Remplacez ci=None par errorbar=None
    plt.title(f"Barplot : Moyenne de la variable {var} par catégorie")
    plt.xlabel("Evolution")
    plt.ylabel(var)
    plt.show()

Deces = dfNoOutliers[dfNoOutliers['Evolution']=='Deces']
Vivant = dfNoOutliers[dfNoOutliers['Evolution']=='Vivant']
for var in num_features:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.distplot(Deces[var],label='Deces')
    plt.axvline(x = np.mean(Deces[var]),label='mean_Deces')
    sns.distplot(Vivant[var],label='Vivant')
    plt.axvline(x = np.mean(Vivant[var]),label='mean_Vivant',color='orange')
    plt.legend()
    plt.show()

"""# **Modélisations**"""

#importation de l'algorithme d'encodage ordinal
from sklearn.preprocessing import OrdinalEncoder

#copy de la base
dfEncoded = dfNoOutliers.copy()
for var in cat_features:
    if var == "Traitement":
        #Encodage
        dfEncoded[var] = dfEncoded[var].map({"Thrombolyse": 1, "Chirurgie": 0})
    elif var == "Evolution":
        #Encodage
        dfEncoded[var] = dfEncoded[var].map({"Vivant": 1, "Deces": 0})
    elif var == "SEXE":
        #Encodage
        dfEncoded[var] = dfEncoded[var].map({"Homme": 1, "Femme": 0})
    else:
        #Encodage
        dfEncoded[var] = dfEncoded[var].map({"NON": 1, "OUI": 0})
#apperçu
dfEncoded.head()

from sklearn.model_selection import train_test_split

y = dfEncoded['Evolution']
X = dfEncoded.drop(columns=['Evolution'],axis=1)

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)
X_train

# Charger les librairies pour l'entrainement de l'arbre
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# initialiser le modèle
classifier = DecisionTreeClassifier(criterion='entropy')

# Entrainement du modèle
classifier = classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)
print('Accuracy score with decision tree:', metrics.accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

pip install pydotplus

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

data = dfEncoded.drop(columns=['Evolution'])

dot_data = StringIO()

from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from io import StringIO

# Génération de la représentation DOT
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=data.columns,
                class_names=['0', '1'])

# Conversion en graphe
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Vérifiez si le graphe est une liste
if isinstance(graph, list):
    graph = graph[0]

# Génération de l'image PNG
Image(graph.create_png())

forest_importances = pd.Series(classifier.feature_importances_,index=data.columns)
forest_importances

from sklearn.neighbors import KNeighborsClassifier

classifier_KNN = KNeighborsClassifier(n_neighbors=3)

X_train.isna().sum(axis = 0)

classifier_KNN.fit(X_train, y_train)

y_pred = classifier_KNN.predict(X_test)
print(f'Accuracy score with KNN classifier: {metrics.accuracy_score(y_test,y_pred)}')

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(f'Accuracy score with KNN classifier: {metrics.accuracy_score(y_test,y_pred)}')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# ces algorithmes utilisent le boosting
classifier_ada_boost = AdaBoostClassifier()
classifier_gradient_boost = GradientBoostingClassifier()

# l'Algorithme du random forest utilise le bagging
classifier_random_forest = RandomForestClassifier()

for model in (classifier_ada_boost, classifier_gradient_boost, classifier_random_forest):
  model.fit(X_train,y_train)

  y_pred = model.predict(X_test)
  print(f'{model} \nAccuracy score with: {metrics.accuracy_score(y_test,y_pred)}')
  cm = confusion_matrix(y_test, y_pred)
  print(cm)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import log_loss
from sklearn import metrics
#import shap
#from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
clf = MLPClassifier(hidden_layer_sizes=(10,20),activation="relu",random_state=3).fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(clf.score(X_test, y_test))

pickle.dump(clf, open('model.pkl','wb'))

print(clf.score(X_train, y_train))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Précision du modèle : à quelle fréquence le classificateur est-il correct ?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Précision du modèle : quel pourcentage de tuples positifs sont étiquetés comme tels ?
print("Precision:",metrics.precision_score(y_test, y_pred))

#Model Recall : quel pourcentage de tuples positifs sont étiquetés comme tels ?
print("Recall:",metrics.recall_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
print(f'''Specificity: {specificity:.3f}''')

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test,y_pred)
#cm = confusion_matrix(y_train, y_test)
sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of Preceptron') # fmt='d' formate les nombres sous forme de chiffres, ce qui signifie des nombres entiers

print(classification_report(y_test,y_pred))

auc = metrics.roc_auc_score(y_test,y_pred)
print(auc)

#define metrics
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(f'AUC : {metrics.roc_auc_score(y_test,y_pred)}')

# Importer les librairies
from sklearn.model_selection import GridSearchCV

# Créer un ensemble contenant nos modèle et nos datasets
tab_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'KNeighbors' : {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1,5,11],
            'p':[1,2,3],
            'weights':['uniform','distance']
        }
    },
    'MLPClassifier' : {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes' : [(10,20),(10,20,10),(10,20,10,10)],
            'activation' : ['relu','logistic'],
            'random_state' : [1, 2, 3]
                  }
    }
}
def test_model_param(params,X,y):
    scores = [] # On va stocker les différents score
    for model_name, mp in params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
    return (pd.DataFrame(scores,columns=['model','best_score','best_params']))

# Testons notre fonction
test_model_param(tab_params,X,y)

"""# Deploiement"""

!pip install fastapi uvicorn

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle pré-entraîné
# Assurez-vous que "model.pkl" est dans le même dossier que ce script
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON envoyées par le frontend
        data = request.json

        # Mapper les données dans l'ordre attendu par le modèle
        # Remplacez "column_names" par l'ordre réel des colonnes utilisé pendant l'entraînement
        column_names = [
            "AGE",
            "SEXE",
            "Premiers Signe - Admission à l'hopital",
            "Admission à l'hopital - Prise en charge medicale",
            "Hypertension Arterielle",
            "Diabete",
            "Cardiopathie",
            "hémiplégie",
            "Paralysie faciale",
            "Aphasie",
            "Hémiparésie",
            "Engagement Cerebral",
            "Inondation Ventriculaire",
            "Traitement",
            "Temps de Suivi après traitement (en jours)",
        ]

        # Créer un vecteur d'entrée
        input_features = [
            data.get(column, 0) for column in column_names
        ]

        # Prétraiter les données si nécessaire (exemple : encodage)
        for i in range(len(input_features)):
            if input_features[i] == "Homme":
              input_features[i] = 1
            elif input_features[i] == "Femme":
              input_features[i] = 0
            elif input_features[i] == "NON":
              input_features[i] = 1
            elif input_features[i] == "OUI":
              input_features[i] = 0
            elif input_features[i] == "Thrombolyse":
              input_features[i] = 1
            else :
              input_features[i] = 0

        # Convertir en tableau NumPy
        input_array = np.array(input_features).reshape(1, -1)

        # Faire une prédiction avec le modèle
        prediction = model.predict(input_array)

        # Retourner la réponse
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Lancer le serveur Flask
    app.run(debug=True)