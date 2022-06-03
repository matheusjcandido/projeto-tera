import pandas as pd 
import streamlit as st
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, plot_precision_recall_curve, classification_report, plot_roc_curve, ConfusionMatrixDisplay
from PIL import Image

image = Image.open('TN_logov.png')

st.image(image, use_column_width=True)

# título
st.title("Desengajamento de clientes do Tecnonutri")

st.write("O Tecnonutri é uma plataforma que conta com uma equipe multidisciplinar, com nutricionistas, psicólogos e treinadores, que auxiliam os clientes em programas de emagrecimento, vida saudável e ganho de massa. O aplicativo contém cardápios, listas de compras, protocolos de jejum intermitente, exercícios, meditações e conteúdos.")

st.header("Esses são os clientes com a maior probabilidade de desengajamento:")

# dataset
df = pd.read_csv('https://raw.githubusercontent.com/matheusjcandido/projeto-tera/main/data_limpo2.csv')

# Definir x e y
x = df.drop(['target', 'id'], axis=1)
y = df['target']

# Definir x_train e y_train, x_test e y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Rodar o modelo

# LGBMClassifier

# lgbm_model = LGBMClassifier(
#     random_state=42,
#     scale_pos_weight = 0.15, # para tratar o balanceamento
#     learning_rate = 0.03,
#     num_leaves = 9,
#     min_child_samples = 7,
#     subsample = 0.73,
#     colsample_bytree = 0.38,
#     n_estimators=300,
#     subsample_freq = 1
# )
# lgbm_model.fit(x_train, y_train)

# lgbm_pred = lgbm_model.predict(x_test)

# lgbm_prob = lgbm_model.predict_proba(x_test)[:,1]
# lgbm_prob_df = pd.DataFrame({'id': x_test.index, 'prob': lgbm_prob})
# lgbm_prob_df = lgbm_prob_df.sort_values('prob', ascending=False)
# lgbm_prob_df = lgbm_prob_df.reset_index(drop=True)
# col1, col2, col3 = st.columns(3)
# col2.write(lgbm_prob_df, use_column_width=True)



# XGBoost

xgb_model = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.01,
    max_depth=7,
    subsample = 0.75,
    colsample_bynode=0.75,
    min_child_weight= 40,
    scale_pos_weight = 0.15 # para balancear o problema de classificação = total negative examples / total positive examples
)
xgb_model.fit(x_train, y_train)

# xgb_pred = xgb_model.predict(x_test)

xgb_prob = xgb_model.predict_proba(x_test)[:,1]

# create a dataframe with 'id' and 'xgb_prob'
xgb_prob_df = pd.DataFrame({'id': x_test.index, 'xgb_prob': xgb_prob})

# ordenar xgb_prob_df por xgb_prob, da maior para a menor
xgb_prob_df = xgb_prob_df.sort_values(by='xgb_prob', ascending=False)
# retirar index
xgb_prob_df = xgb_prob_df.reset_index(drop=True)

#criando 3 colunas 
col1, col2, col3 = st.columns(3)
#inserindo na coluna 2
col2.write(xgb_prob_df, use_column_width=True)
#st.write(xgb_prob_df)
