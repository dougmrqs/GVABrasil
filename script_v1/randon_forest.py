import pandas as pd

base = pd.read_csv('arquivocsv.csv')
'''
formato do csv que testei:
4 colunas como previsores
1 coluna como classe
> colunas previssores as keywords que mais aparecem
> classe: 1 (1 se é relevante) e 0 se não for
'''
               
#divide o csv entre previsores e classe
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

#substitui os valores nulos pela média
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

#normaliza a escala para todos os campos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#divide a base entre base de teste e base de treinamento
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#cria o classificador
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#cria a matriz de confusão para verificar o índice de acerto
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)