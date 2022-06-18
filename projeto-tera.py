import pandas as pd 
import streamlit as st
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from PIL import Image

image = Image.open('tera_banner.png')

st.image(image, use_column_width=True)

# título
st.title("Desengajamento de clientes de um app de saúde 📱🍲🏃‍♀️🧘‍♂️")

st.write("O app de saúde estudado é uma plataforma que conta com uma equipe multidisciplinar, com nutricionistas, psicólogos e treinadores, que auxiliam os clientes em programas de emagrecimento, vida saudável e ganho de massa. O aplicativo contém cardápios, listas de compras, protocolos de jejum intermitente, exercícios, meditações e conteúdos.")

st.write("Nesta página, serão apresentadas informações sobre a atividade dos usuários de acordo com características específicas e, ao final, serão expostos os resultados de um modelo de previsão que indica quais os usuários com a maior probabilidade de desengajarem.")

# dataset
df = pd.read_csv('https://raw.githubusercontent.com/matheusjcandido/projeto-tera/main/data_limpo2.csv')


#### POR CARACTERÍSTICAS DO USUÁRIO ####
st.header("Cohorts:")
st.write("Serão mostrados gráficos que revelam a quantidade de dias que os usuários entram no app, agrupados de acordo com características específicas.")

st.write("No geral, os usuários entram no aplicativo em 19 dias (em média) diferentes. O heatmap a seguir traz um cohort geral e mensal, de outubro de 2019 a dezembro de 2021, o que será detalhado por grupos específicos nos cohorts seguintes.")



st.image('images/1_cohort_mensal.png', use_column_width=True)


st.subheader("Cohorts de acordo com características do usuário:")


select = st.selectbox('Escolha um cohort para visualização:', ('Escolha', 'Faixa etária', 'Gênero', 'Faixas de peso desejado', 'Objetivo do usuário'))
if select == 'Escolha':
    pass
elif select == 'Faixa etária':
    st.image('images/3_faixa_etaria.png', use_column_width=True)
    st.caption("Os usuários de faixas etárias mais elevadas entram, em média, mais dias no aplicativo do que os usuários mais jovens. Isso também é refletido no cohort de engajamento ao longo dos dias: no 20º dia, temos mais de 20% dos usuários das faixas dos 40 aos 70 anos, enquanto que para os usuários de 20 a 30 anos esse percentual é de apenas 12%.")

elif select == 'Gênero':
    st.image('images/2_genero.png', use_column_width=True)
    st.caption("Os usuários do gênero masculino entram no app 15 dias (em média), enquanto as mulheres entram em média 20 dias. Isso também é refletido no cohort de engajamento ao longo dos dias: no 20º dia, temos 18% dos usuários do gênero feminino e apenas 13% do masculino.")

elif select == 'Faixas de peso desejado':
    st.image('images/5_peso.png', use_column_width=True)
    st.caption("Os usuários que estão em faixas mais elevadas, seja de emagrecimento ou de ganho de massa, entram em média menos dias no aplicativo. Isso também é refletido no cohort de engajamento ao longo dos dias: no 20º dia, temos 20% dos usuários da faixa de emagrecimento de até 10 kg, enquanto que para os usuários que querem emagrecer entre 30 e 40 kg ou ganhar mais de 10 kg esse percentual é de apenas 15% e 10%, respectivamente.")

elif select == 'Objetivo do usuário':
    st.image('images/4_objetivo.png', use_column_width=True)
    st.caption("Os usuários que querem perder peso são os mais representativos e entram em média 21 dias no aplicativo. Observando o cohort de engajamento, no 20º dia temos 19% dos usuários que querem perder peso e apenas 9% dos que querem ganhar massa (segunda maior categoria).")


### POR PARTICIPAÇÃO NO APLICATIVO ###
st.subheader("Cohorts dos usuários de acordo com a utilização do aplicativo:")

option = st.selectbox('Escolha um cohort para visualização:', ('Escolha', 'Plataforma', 'Inscrição', 'Número de programas', 'Número de posts', 'Análise nutricional', 'Treinos', 'Meditações', 'Diário de refeições'))
if option == 'Escolha':
    pass
elif option == 'Plataforma':
    st.image('images/7_platform.png', use_column_width=True)
    st.caption("Os usuários de Android e iOS têm um comportamento parecido em relação à quantidade de dias que entraram no aplicativo. No cohort de engajamento vemos que no 20º dia, temos 17% e 19% dos usuários de Android e iOS, respectivamente.")

elif option == 'Inscrição':
    st.image('images/8_is_program_enrolled.png', use_column_width=True)
    st.caption("No cohort de engajamento vemos que no 20º dia temos 21% dos usuários que estão inscritos em programas. Para os usuários que não estão inscritos em programas, esse percentual é de 15%.")

elif option == 'Número de programas':
    st.image('images/9_n_programs.png', use_column_width=True)
    st.caption("Os usuários que se inscrevem em mais programas entram em média mais dias no aplicativo. A quantidade de programas em que o usuário se inscreveu poderia ser apenas uma proxy do tempo que o usuário permaneceu no aplicativo. O que é interessante notar no cohort de engajamento é que, desde o início, usuários que vão se inscrever em mais programas ao longo de sua vida no aplicativo engajam mais. No 20º dia, temos quase 35% dos usuários que se inscreveram em quatro ou cinco programas. Esse número cai para apenas 12% para aqueles que não se inscreveram ou que se escreveram em apenas um programa.")

elif option == 'Número de posts':
    st.image('images/10_posts', use_column_width=True)
    st.caption("Pela visão do cohort também fica bastante evidente a diferença entre o engajamento dos usuários que fizeram um post e daqueles que não o fizeram. No 20º dia, temos apenas 13% dos usuários que nunca fizeram um post e 41% daqueles que fizeram.")

elif option == 'Análise nutricional':
    st.image('images/11_nutritional.png', use_column_width=True)
    st.caption("Pela visão do cohort fica bastante evidente a diferença entre o engajamento dos usuários que fizeram análise nutricional e daqueles que não o fizeram. No 20º dia, temos apenas 17% dos usuários que nunca fizeram uma análise e 47% daqueles que fizeram.")

elif option == 'Treinos':
    st.image('images/12_treino.png', use_column_width=True)
    st.caption("O cohort mostra a diferença entre o engajamento dos usuários que realizaram treinos e daqueles que não fizeram. No 20º dia, temos apenas 15% dos usuários que nunca fizeram um treino e 36% daqueles que fizeram.")

elif option == 'Meditações':
    st.image('images/13_meditation.png', use_column_width=True)
    st.caption("O cohort indica a diferença entre o engajamento dos usuários que realizaram meditações e daqueles que não o fizeram. No 20º dia, temos apenas 16% dos usuários que nunca fizeram uma meditação e 40% daqueles que fizeram.")

elif option == 'Diário de refeições':
    st.image('images/14_diario.png', use_column_width=True)
    st.caption("O cohort deixa evidente a diferença entre o engajamento dos usuários que registraram refeições e daqueles que não fizeram. No 20º dia, temos apenas 3% dos usuários que nunca fizeram um registro e 21% daqueles que fizeram.")

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
    learning_rate = 0.014,
    num_leaves = 98,
    min_child_samples = 85,
    subsample = 0.81,
    colsample_bytree = 0.92,
    n_estimators=405,
    subsample_freq = 3
)
lgbm_model.fit(x_train, y_train)

# lgbm_pred = lgbm_model.predict(x_test)

lgbm_prob = lgbm_model.predict_proba(x_test)[:,1]

lgbm_prob_df = pd.DataFrame({'ID': x_test.index, 'prob': lgbm_prob})


lgbm_prob_df = lgbm_prob_df.sort_values('prob', ascending=False)
lgbm_prob_df = lgbm_prob_df.reset_index(drop=True)
lgbm_prob_df_head = lgbm_prob_df.head(200)

if st.button('Rodar modelo'):
    col1, col2, col3 = st.columns(3)
    col2.write(lgbm_prob_df_head.iloc[:, 0], use_column_width=True)


