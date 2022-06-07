import pandas as pd 
import streamlit as st
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, plot_precision_recall_curve, classification_report, plot_roc_curve, ConfusionMatrixDisplay
from PIL import Image
import plotly.figure_factory as ff
import plotly.express as px

image = Image.open('tera_banner.png')

st.image(image, use_column_width=True)

# título
st.title("Desengajamento de clientes de um app de saúde 📱🍲🏃‍♀️🧘‍♂️")

st.write("O app de saúde estudado é uma plataforma que conta com uma equipe multidisciplinar, com nutricionistas, psicólogos e treinadores, que auxiliam os clientes em programas de emagrecimento, vida saudável e ganho de massa. O aplicativo contém cardápios, listas de compras, protocolos de jejum intermitente, exercícios, meditações e conteúdos.")

st.write("Tendo impactado a vida de milhões de pessoas, atualmente conta com 30 mil instalações e 60 mil usuários únicos por semana.")

st.write("Nesta página, serão apresentadas informações sobre a atividade dos usuários de acordo com características específicas e, ao final, serão expostos os resultados de um modelo de previsão que indica quais os usuários com a maior probabilidade de desengajarem.")

# dataset
df = pd.read_csv('https://github.com/matheusjcandido/projeto-tera/blob/main/data_limpo2.csv', low_memory=False, header=None, sep='delimiter')
# df_cohorts = pd.read_csv('https://drive.google.com/file/d/1w99zwweds2buuJDkuHFh-t-wZIMxtl17/view?usp=sharing')
df_cohorts = pd.read_csv('https://www.dropbox.com/s/t7dplc9h8353e60/cohorts.csv?dl=1', header = None, sep = 'delimiter', low_memory=False)
# df_diff_days = df_cohorts.groupby(['user_id'])['diff_days'].max().reset_index(name='max_diff_days')
# #Histograma dos usuários de acordo com o último dia que entraram no aplicativo
# hist_diff_days = px.bar(df_diff_days['max_diff_days'].value_counts(normalize=True))
# st.plotly_chart(hist_diff_days)
# st.caption("Histograma dos usuários de acordo com o último dia em que entraram no aplicativo.")

# df_cohort = df_cohorts['diff_days'].value_counts().rename_axis('days').reset_index(name='counts')
# #Adiciona coluna de frequência: dos usuários que estavam no dia 0, quantos voltaram em cada dia?
# df_cohort['frequencia'] = df_cohort['counts']/df_cohort.iloc[0,1]*100
# #Histograma dos dias e frequência de usuários
# hist_frequency = px.bar(df_cohort[['days','frequencia']], x ='days', y='frequencia')
# st.plotly_chart(hist_frequency)

# df_cohorts['created_month_year'] = pd.to_datetime(df_cohorts['created_at']).dt.to_period('M')

# #Cria dataframe com os dias, mês e anos de criação e a quantidade de usuários em cada um deles
# df_cohort_dates = df_cohorts.groupby(['diff_days','created_month_year'])['user_id'].count().reset_index(name='usuários')
# df_cohort_dates = df_cohort_dates[df_cohort_dates['diff_days']<21]

# #Pivotear dataframe para termos os dias nas colunas
# cohort_pivot_dates = df_cohort_dates.pivot_table(index = 'created_month_year',
#                                      columns = 'diff_days',
#                                      values = 'usuários')


#### POR CARACTERÍSTICAS DO USUÁRIO ####
st.header("Cohorts:")
st.write("Serão mostrados gráficos que revelam a quantidade de dias que os usuários entram no app, agrupados de acordo com características específicas.")

st.write("No geral, os usuários entram no aplicativo 19 dias (em média). O boxplot a seguir traz a distribuição geral dos usuários, o que será detalhado por grupos específicos nos cohorts posteriores.")

#### ---------------- GRÁFICOS ---------------- ####
df_cohorts = df_cohorts.drop(df_cohorts[df_cohorts['censored'] == 1].index)
df_cohorts['rank_user_by_date'] = df_cohorts.groupby('user_id')['date'].rank(method='first')
df_rank_user_adj = df_cohorts.groupby(['user_id'])['rank_user_by_date'].max().reset_index(name='max_day')
rank_user_box = px.box(df_rank_user_adj, x = 'max_day', points = 'all')
st.plotly_chart(rank_user_box)
st.caption("Quantidade de dias que os usuários entraram no app.")



st.subheader("Cohorts de acordo com características do usuário:")


select = st.selectbox('Escolha um cohort para visualização:', ('Escolha', 'Faixa etária', 'Gênero', 'Faixas de peso desejado', 'Objetivo do usuário'))
if select == 'Escolha':
    pass
elif select == 'Faixa etária':
    #Cria faixas etárias
    df_cohorts['faixa_etaria'] = df_cohorts['client_age']

    df_cohorts.loc[df_cohorts['client_age'] < 20,'faixa_etaria'] = '0 a 20 anos'
    df_cohorts.loc[(df_cohorts['client_age'] >= 20) & (df_cohorts['client_age'] < 30),'faixa_etaria'] = '20 a 30 anos'
    df_cohorts.loc[(df_cohorts['client_age'] >= 30) & (df_cohorts['client_age'] < 40),'faixa_etaria'] = '30 a 40 anos'
    df_cohorts.loc[(df_cohorts['client_age'] >= 40) & (df_cohorts['client_age'] < 50),'faixa_etaria'] = '40 a 50 anos'
    df_cohorts.loc[(df_cohorts['client_age'] >= 50) & (df_cohorts['client_age'] < 60),'faixa_etaria'] = '50 a 60 anos'
    df_cohorts.loc[(df_cohorts['client_age'] >= 60) & (df_cohorts['client_age'] < 70),'faixa_etaria'] = '60 a 70 anos'
    df_cohorts.loc[df_cohorts['client_age'] >= 70,'faixa_etaria'] = '70 ou mais'

    #Cria dataframe com os usuários, faixa etária e a quantidade de dias em que entraram no app
    df_rank_user_age = df_cohorts.groupby(['user_id','faixa_etaria'])['rank_user_by_date'].max().reset_index(name='max_day')

    #Boxplots da quantidade de dias que os usuários entraram no app por faixa etária
    age_boxplot = px.box(df_rank_user_age, x = 'max_day', color = 'faixa_etaria', height = 350, width = 900, category_orders = {'faixa_etaria': ['0 a 20 anos','20 a 30 anos','30 a 40 anos','40 a 50 anos','50 a 60 anos','60 a 70 anos','70 ou mais']})
    st.plotly_chart(age_boxplot)
    st.caption("")

elif select == 'Gênero':
    #Cria dataframe com os usuários, gênero e a quantidade de dias em que entraram no app
    df_rank_user_gender = df_cohorts.groupby(['user_id','gender'])['rank_user_by_date'].max().reset_index(name='max_day')
    df_rank_user_gender.groupby(['gender'])['max_day'].describe()

    #Boxplots da quantidade de dias que os usuários entraram no app por gênero
    gender_boxplot = px.box(df_rank_user_gender, x = 'max_day', color = 'gender', height = 350, width = 900)
    st.plotly_chart(gender_boxplot)
    st.caption("Os usuários do gênero masculino entram no app 15 dias (em média), enquanto as mulheres entram em média 20 dias.")

elif select == 'Faixas de peso desejado':
    #Criação de nova variável com a diferença entre o objetivo de peso e o peso atual do usuário
    df_cohorts['delta_weight'] = df_cohorts['ideal_weight'] - df_cohorts['weight']
    #Cria faixas de peso
    df_cohorts['faixa_peso'] = df_cohorts['delta_weight']

    df_cohorts.loc[df_cohorts['delta_weight'] < -50,'faixa_peso'] = 'a) -50 kg ou mais'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= -50) & (df_cohorts['delta_weight'] < -40),'faixa_peso'] = 'b) -50 a -40 kg'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= -40) & (df_cohorts['delta_weight'] < -30),'faixa_peso'] = 'c) -40 a -30 kg'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= -30) & (df_cohorts['delta_weight'] < -20),'faixa_peso'] = 'd) -30 a -20 kg'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= -20) & (df_cohorts['delta_weight'] < -10),'faixa_peso'] = 'e) -20 a -10 kg'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= -10) & (df_cohorts['delta_weight'] < 0),'faixa_peso'] = 'f) -10 a 0 kg'
    df_cohorts.loc[(df_cohorts['delta_weight'] >= 0) & (df_cohorts['delta_weight'] < 10),'faixa_peso'] = 'g) 0 a 10 kg'
    df_cohorts.loc[df_cohorts['delta_weight'] >= 10,'faixa_peso'] = 'h) 10 kg ou mais'
    #Cria dataframe com os usuários, faixa etária e a quantidade de dias em que entraram no app
    df_rank_user_weight = df_cohorts.groupby(['user_id','faixa_peso'])['rank_user_by_date'].max().reset_index(name='max_day')
    #Boxplots da quantidade de dias que os usuários entraram no app por faixa etária
    peso_box = px.box(df_rank_user_weight, x = 'max_day', color = 'faixa_peso', height = 350, width = 900, category_orders={'faixa_peso': ['a) -50 kg ou mais','b) -50 a -40 kg','c) -40 a -30 kg','d) -30 a -20 kg','e) -20 a -10 kg','f) -10 a 0 kg','g) 0 a 10 kg','h) 10 kg ou mais']})
    st.plotly_chart(peso_box)
    st.caption("As faixas etárias dos 20 aos 30 anos e dos 30 aos 40 anos são as mais representativas. Para essas faixas, a média de entrada no aplicativo foi de 13 e 18 dias, respectivamente. Embora sejam menos representativas, as faixas dos 50 aos 60 anos e dos 60 aos 70 anos entraram em média 26 e 27 dias no aplicativo.")

elif select == 'Objetivo do usuário':
    #Cria campo de objetivo
    df_cohorts['general_goal'] = df_cohorts['general_goal_Perder peso']

    df_cohorts.loc[df_cohorts['general_goal_Perder peso'] == 1,'general_goal'] = 'Perder peso'
    df_cohorts.loc[df_cohorts['general_goal_Ganhar massa'] == 1,'general_goal'] = 'Ganhar massa'
    df_cohorts.loc[df_cohorts['general_goal_Melhorar alimentação'] == 1,'general_goal'] = 'Melhorar alimentação'
    #Cria dataframe com os usuários, objetivo e a quantidade de dias em que entraram no app
    df_rank_user_goal = df_cohorts.groupby(['user_id','general_goal'])['rank_user_by_date'].max().reset_index(name='max_day')
    df_rank_user_goal = df_rank_user_goal[df_rank_user_goal['general_goal'] != 0]
    objective_box = px.box(df_rank_user_goal, x = 'max_day', color = 'general_goal', height = 350, width = 900)
    st.plotly_chart(objective_box)
    st.caption("")


### POR PARTICIPAÇÃO NO APLICATIVO ###
st.subheader("Cohorts dos usuários de acordo com a utilização do aplicativo:")

option = st.selectbox('Escolha um cohort para visualização:', ('Escolha', 'Plataforma', 'Número de programas', 'Análise nutricional', 'Treinos', 'Meditações', 'Diário de refeições'))
if option == 'Escolha':
    pass
elif option == 'Plataforma':
    #Cria campo de plataforma
    df_cohorts['platform'] = df_cohorts['platform_android']

    df_cohorts.loc[df_cohorts['platform_android'] == 1,'platform'] = 'Android'
    df_cohorts.loc[df_cohorts['platform_ios'] == 1,'platform'] = 'iOS'
    df_cohorts.loc[df_cohorts['platform_web'] == 1,'platform'] = 'Web'

    #Cria dataframe com os usuários, plataforma e a quantidade de dias em que entraram no app
    df_rank_user_platform = df_cohorts.groupby(['user_id','platform'])['rank_user_by_date'].max().reset_index(name='max_day')

    #Boxplots da quantidade de dias que os usuários entraram no app por plataforma
    platform_boxplot = px.box(df_rank_user_platform, x = 'max_day', color = 'platform', height = 350, width = 900)
    st.plotly_chart(platform_boxplot)
    st.caption("")

elif option == 'Número de programas':
    #Cria faixas de quantidade de programas
    df_cohorts['faixa_programa'] = df_cohorts['programs_enrolled_count']

    df_cohorts.loc[df_cohorts['programs_enrolled_count'] < 2,'faixa_programa'] = '0 a 2'
    df_cohorts.loc[(df_cohorts['programs_enrolled_count'] >= 2) & (df_cohorts['programs_enrolled_count'] < 4),'faixa_programa'] = '2 a 4'
    df_cohorts.loc[(df_cohorts['programs_enrolled_count'] >= 4) & (df_cohorts['programs_enrolled_count'] < 6),'faixa_programa'] = '4 a 6'
    df_cohorts.loc[(df_cohorts['programs_enrolled_count'] >= 6) & (df_cohorts['programs_enrolled_count'] < 8),'faixa_programa'] = '6 a 8'
    df_cohorts.loc[(df_cohorts['programs_enrolled_count'] >= 8) & (df_cohorts['programs_enrolled_count'] < 10),'faixa_programa'] = '8 a 10'
    df_cohorts.loc[df_cohorts['programs_enrolled_count'] >= 10,'faixa_programa'] = 'mais de 10'

    #Cria dataframe com os usuários, faixa de programas e a quantidade de dias em que entraram no app
    df_rank_user_program_count = df_cohorts.groupby(['user_id','faixa_programa'])['rank_user_by_date'].max().reset_index(name='max_day')

    #Boxplots da quantidade de dias que os usuários entraram no app por faixas de programas
    programs_quantity_box = px.box(df_rank_user_program_count, x = 'max_day', color = 'faixa_programa', height = 350, width = 900, category_orders = {'faixa_programa': ['0 a 2','2 a 4','4 a 6','6 a 8','8 a 10','mais de 10']})
    st.plotly_chart(programs_quantity_box)
    st.caption("")

elif option == 'Análise nutricional':
    #Cria campo de análise nutricional
    df_cohorts['has_nutritional_analysis'] = df_cohorts['nutritional_analyses_count']

    df_cohorts.loc[df_cohorts['nutritional_analyses_count'] > 0,'has_nutritional_analysis'] = 'Com análise nutricional'
    df_cohorts.loc[df_cohorts['nutritional_analyses_count'] == 0,'has_nutritional_analysis'] = 'Sem análise nutricional'
    df_rank_user_nutri = df_cohorts.groupby(['user_id','has_nutritional_analysis'])['rank_user_by_date'].max().reset_index(name='max_day')
    #Boxplots da quantidade de dias que os usuários entraram no app por realização de análise nutricional
    nutri_analysis_box = px.box(df_rank_user_nutri, x = 'max_day', color = 'has_nutritional_analysis', height = 350, width = 900)
    st.plotly_chart(nutri_analysis_box)
    st.caption("")

elif option == 'Treinos':
    #Cria campo de treino
    df_cohorts['has_training'] = df_cohorts['trainings_count']

    df_cohorts.loc[df_cohorts['trainings_count'] > 0,'has_training'] = 'Com treino'
    df_cohorts.loc[df_cohorts['trainings_count'] == 0,'has_training'] = 'Sem treino'
    #Cria dataframe com os usuários, se fizeram treinos e a quantidade de dias em que entraram no app
    df_rank_user_training = df_cohorts.groupby(['user_id','has_training'])['rank_user_by_date'].max().reset_index(name='max_day')
    #Boxplots da quantidade de dias que os usuários entraram no app por realização de treino
    training_box = px.box(df_rank_user_training, x = 'max_day', color = 'has_training', height = 350, width = 900)
    st.plotly_chart(training_box)
    st.caption("")

elif option == 'Meditações':
    #Cria campo de meditações
    df_cohorts['has_meditation'] = df_cohorts['meditations_count']

    df_cohorts.loc[df_cohorts['meditations_count'] > 0,'has_meditation'] = 'Com meditação'
    df_cohorts.loc[df_cohorts['meditations_count'] == 0,'has_meditation'] = 'Sem meditação'
    #Cria dataframe com os usuários, se fizeram meditações e a quantidade de dias em que entraram no app
    df_rank_user_meditation = df_cohorts.groupby(['user_id','has_meditation'])['rank_user_by_date'].max().reset_index(name='max_day')
    #Boxplots da quantidade de dias que os usuários entraram no app por realização de meditação
    meditations_box = px.box(df_rank_user_meditation, x = 'max_day', color = 'has_meditation', height = 350, width = 900)
    st.plotly_chart(meditations_box)
    st.caption("")

elif option == 'Diário de refeições':
    #Cria campo de diário de refeições
    df_cohorts['has_meal'] = df_cohorts['meals_count_adj']

    df_cohorts.loc[df_cohorts['meals_count_adj'] > 0,'has_meal'] = 'Com diário'
    df_cohorts.loc[df_cohorts['meals_count_adj'] == 0,'has_meal'] = 'Sem diário'
    #Cria dataframe com os usuários, se fizeram meditações e a quantidade de dias em que entraram no app
    df_rank_user_meal = df_cohorts.groupby(['user_id','has_meal'])['rank_user_by_date'].max().reset_index(name='max_day')
    #Boxplots da quantidade de dias que os usuários entraram no app por anotação de refeição
    meal_box = px.box(df_rank_user_meal, x = 'max_day', color = 'has_meal', height = 350, width = 900)
    st.plotly_chart(meal_box)
    st.caption("")

st.header("Modelo de predição:")

st.write("O grande objetivo do trabalho foi construir um modelo de predição para verificar quais os clientes com as maiores probabilidades de desengajamento do aplicativo, o que possibilitaria a execução de ações direcionadas para prevenir o churn.")
st.write("O modelo utilizado para predição é o LightGBM, que é construído a partir de algorimos de árvores de decisão. Aqui serão mostrados os IDs dos 200 clientes com maior probabilidade de desengajamento, ordenados da maior para a menor probabilidade.")


#### ---------------- LIGHTGBM  ---------------- ####

# Json special characters fix
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Definir x e y
x = df.drop(['target', 'id'], axis=1)
y = df['target']

# Definir x_train e y_train, x_test e y_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Rodar o modelo
st.subheader("Esses são os clientes com a maior probabilidade de desengajamento:")

# LGBMClassifier

lgbm_model = LGBMClassifier(
    random_state=42,
    scale_pos_weight = 0.15, # para tratar o balanceamento
    learning_rate = 0.008,
    num_leaves = 13,
    min_child_samples = 65,
    subsample = 0.86,
    colsample_bytree = 0.53,
    n_estimators=973,
    subsample_freq = 7
)
lgbm_model.fit(x_train, y_train)

lgbm_pred = lgbm_model.predict(x_test)

lgbm_prob = lgbm_model.predict_proba(x_test)[:,1]

# selecionar os clientes que apresentam predição igual a 1
lgbm_pred_1 = lgbm_model.predict(x_test)[lgbm_pred == 1]

lgbm_prob_df = pd.DataFrame({'ID': x_test.index, 'prob': lgbm_prob})


lgbm_prob_df = lgbm_prob_df.sort_values('prob', ascending=False)
lgbm_prob_df = lgbm_prob_df.reset_index(drop=True)
lgbm_prob_df_head = lgbm_prob_df.head(200)

if st.button('Rodar modelo'):
    col1, col2, col3 = st.columns(3)
    col2.write(lgbm_prob_df_head.iloc[:, 0], use_column_width=True)



# XGBoost

# xgb_model = XGBClassifier(
#     n_estimators=500, 
#     learning_rate=0.01,
#     max_depth=7,
#     subsample = 0.75,
#     colsample_bynode=0.75,
#     min_child_weight= 40,
#     scale_pos_weight = 0.15 # para balancear o problema de classificação = total negative examples / total positive examples
# )
# xgb_model.fit(x_train, y_train)

# # xgb_pred = xgb_model.predict(x_test)

# xgb_prob = xgb_model.predict_proba(x_test)[:,1]

# # create a dataframe with 'id' and 'xgb_prob'
# xgb_prob_df = pd.DataFrame({'id': x_test.index, 'xgb_prob': xgb_prob})

# # ordenar xgb_prob_df por xgb_prob, da maior para a menor
# xgb_prob_df = xgb_prob_df.sort_values(by='xgb_prob', ascending=False)
# # retirar index
# xgb_prob_df = xgb_prob_df.reset_index(drop=True)
# xgb_prob_df_head = xgb_prob_df.head(200)

# #criando 3 colunas 
# col1, col2, col3 = st.columns(3)
# #inserindo na coluna 2
# col2.write(xgb_prob_df_head, use_column_width=True)
