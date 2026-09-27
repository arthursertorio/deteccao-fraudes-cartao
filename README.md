# ============================================================
# PROJETO: DETECÇÃO DE FRAUDES EM TRANSAÇÕES DE CARTÃO
# ============================================================

# Caso esteja usando Google Colab e alguma biblioteca não esteja
# instalada, execute antes:
#
# !pip install xgboost imbalanced-learn shap


# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

import shap


# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print("=" * 60)
print("DATASET CARREGADO")
print("=" * 60)


# ============================================================
# 3. EXPLORAÇÃO INICIAL
# ============================================================

print("\n===== PRIMEIRAS LINHAS =====")
display(df.head())

print("\n===== DIMENSÕES =====")
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n===== INFORMAÇÕES =====")
df.info()

print("\n===== VALORES AUSENTES =====")
print(df.isnull().sum().sum())

print("\n===== ESTATÍSTICAS =====")
display(df.describe())


# ============================================================
# 4. ANÁLISE DO DESBALANCEAMENTO
# ============================================================

print("\n===== QUANTIDADE DE CADA CLASSE =====")

contagem_classes = df["Class"].value_counts()

print(contagem_classes)

proporcao = df["Class"].value_counts(normalize=True) * 100

print("\n===== PROPORÇÃO =====")
print(f"Transações normais: {proporcao[0]:.2f}%")
print(f"Fraudes: {proporcao[1]:.2f}%")


# Gráfico das classes

plt.figure(figsize=(6, 4))

df["Class"].value_counts().plot(kind="bar")

plt.title("Distribuição das Classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.xticks([0, 1], ["Normal", "Fraude"], rotation=0)

plt.show()


# ============================================================
# 5. ANÁLISE DO VALOR DAS TRANSAÇÕES
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df[df["Class"] == 0]["Amount"],
    bins=50,
    alpha=0.7,
    label="Normal"
)

plt.hist(
    df[df["Class"] == 1]["Amount"],
    bins=50,
    alpha=0.7,
    label="Fraude"
)

plt.xlabel("Valor da transação")
plt.ylabel("Frequência")
plt.title("Distribuição dos valores das transações")
plt.legend()

plt.show()


# ============================================================
# 6. PREPARAÇÃO DOS DADOS
# ============================================================

# Criar variável logarítmica para Amount

df["Amount_log"] = np.log1p(df["Amount"])


# Separar variáveis explicativas e variável alvo

X = df.drop("Class", axis=1)
y = df["Class"]


# Remover Amount original

X = X.drop("Amount", axis=1)


# ============================================================
# 7. DIVISÃO ENTRE TREINO, VALIDAÇÃO E TESTE
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\n===== TAMANHO DOS CONJUNTOS =====")
print(f"Treino: {X_train.shape}")
print(f"Validação: {X_val.shape}")
print(f"Teste: {X_test.shape}")


# ============================================================
# 8. PADRONIZAÇÃO
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. FUNÇÃO PARA AVALIAÇÃO DOS MODELOS
# ============================================================

def avaliar_modelo(nome, modelo, X_avaliacao, y_avaliacao):

    previsoes = modelo.predict(X_avaliacao)

    precision = precision_score(
        y_avaliacao,
        previsoes,
        zero_division=0
    )

    recall = recall_score(
        y_avaliacao,
        previsoes,
        zero_division=0
    )

    f1 = f1_score(
        y_avaliacao,
        previsoes,
        zero_division=0
    )

    probabilidades = modelo.predict_proba(X_avaliacao)[:, 1]

    roc_auc = roc_auc_score(
        y_avaliacao,
        probabilidades
    )

    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    print(
        classification_report(
            y_avaliacao,
            previsoes,
            target_names=["Normal", "Fraude"],
            zero_division=0
        )
    )

    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "Modelo": nome,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 10. REGRESSÃO LOGÍSTICA
# ============================================================

modelo_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

modelo_lr.fit(
    X_train_scaled,
    y_train
)

resultado_lr = avaliar_modelo(
    "Regressão Logística",
    modelo_lr,
    X_test_scaled,
    y_test
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

modelo_rf.fit(
    X_train,
    y_train
)

resultado_rf = avaliar_modelo(
    "Random Forest",
    modelo_rf,
    X_test,
    y_test
)


# ============================================================
# 12. XGBOOST
# ============================================================

negativos = (y_train == 0).sum()
positivos = (y_train == 1).sum()

peso_fraude = negativos / positivos

print("\nPeso da classe fraude:")
print(peso_fraude)

modelo_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=peso_fraude,
    random_state=42,
    eval_metric="logloss"
)

modelo_xgb.fit(
    X_train,
    y_train
)

resultado_xgb = avaliar_modelo(
    "XGBoost",
    modelo_xgb,
    X_test,
    y_test
)


# ============================================================
# 13. COMPARAÇÃO DOS MODELOS
# ============================================================

resultados = pd.DataFrame([
    resultado_lr,
    resultado_rf,
    resultado_xgb
])

print("\n===== COMPARAÇÃO DOS MODELOS =====")

display(
    resultados.sort_values(
        by="Recall",
        ascending=False
    )
)


# ============================================================
# 14. OVERSAMPLING
# ============================================================

oversampler = RandomOverSampler(
    random_state=42
)

X_train_over, y_train_over = oversampler.fit_resample(
    X_train,
    y_train
)

print("\n===== OVERSAMPLING =====")
print(y_train_over.value_counts())

modelo_over = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

modelo_over.fit(
    X_train_over,
    y_train_over
)

resultado_over = avaliar_modelo(
    "Random Forest + Oversampling",
    modelo_over,
    X_test,
    y_test
)


# ============================================================
# 15. UNDERSAMPLING
# ============================================================

undersampler = RandomUnderSampler(
    random_state=42
)

X_train_under, y_train_under = undersampler.fit_resample(
    X_train,
    y_train
)

print("\n===== UNDERSAMPLING =====")
print(y_train_under.value_counts())

modelo_under = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

modelo_under.fit(
    X_train_under,
    y_train_under
)

resultado_under = avaliar_modelo(
    "Random Forest + Undersampling",
    modelo_under,
    X_test,
    y_test
)


# ============================================================
# 16. COMPARAÇÃO DO BALANCEAMENTO
# ============================================================

resultados_balanceamento = pd.DataFrame([
    resultado_rf,
    resultado_over,
    resultado_under
])

print("\n===== COMPARAÇÃO DO BALANCEAMENTO =====")

display(
    resultados_balanceamento.sort_values(
        by="Recall",
        ascending=False
    )
)


# ============================================================
# 17. AJUSTE DO LIMIAR
# ============================================================

prob_val = modelo_xgb.predict_proba(
    X_val
)[:, 1]

limiares = np.arange(
    0.05,
    0.96,
    0.05
)

resultados_limiar = []

for limiar in limiares:

    pred = (
        prob_val >= limiar
    ).astype(int)

    resultados_limiar.append({

        "Limiar": limiar,

        "Precision": precision_score(
            y_val,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_val,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_val,
            pred,
            zero_division=0
        )
    })

df_limiar = pd.DataFrame(
    resultados_limiar
)

print("\n===== EFEITO DO LIMIAR =====")

display(df_limiar)


# ============================================================
# 18. ESCOLHA DO LIMIAR
# ============================================================

melhor_linha = df_limiar.loc[
    df_limiar["F1"].idxmax()
]

melhor_limiar = melhor_linha["Limiar"]

print(
    f"\nLimiar escolhido: {melhor_limiar:.2f}"
)


# ============================================================
# 19. AVALIAÇÃO FINAL COM O LIMIAR
# ============================================================

prob_test = modelo_xgb.predict_proba(
    X_test
)[:, 1]

pred_test_limiar = (
    prob_test >= melhor_limiar
).astype(int)

print("\n===== RESULTADO FINAL =====")

print(
    classification_report(
        y_test,
        pred_test_limiar,
        target_names=["Normal", "Fraude"],
        zero_division=0
    )
)


# ============================================================
# 20. GRÁFICO DO LIMIAR
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df_limiar["Limiar"],
    df_limiar["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    df_limiar["Limiar"],
    df_limiar["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    df_limiar["Limiar"],
    df_limiar["F1"],
    marker="o",
    label="F1"
)

plt.xlabel("Limiar")
plt.ylabel("Métrica")
plt.title("Precision, Recall e F1 por Limiar")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# 21. MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(
    y_test,
    pred_test_limiar
)

print("\n===== MATRIZ DE CONFUSÃO =====")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraude"]
).plot()

plt.title("Matriz de Confusão - XGBoost")
plt.show()


# ============================================================
# 22. CURVA ROC
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    prob_test
)

plt.title("Curva ROC - XGBoost")
plt.show()


# ============================================================
# 23. CURVA PRECISION-RECALL
# ============================================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    prob_test
)

plt.title("Curva Precision-Recall - XGBoost")
plt.show()


# ============================================================
# 24. IMPORTÂNCIA DAS VARIÁVEIS - RANDOM FOREST
# ============================================================

importancias_rf = pd.Series(
    modelo_rf.feature_importances_,
    index=X_train.columns
)

importancias_rf = importancias_rf.sort_values(
    ascending=False
).head(15)

plt.figure(figsize=(8, 6))

importancias_rf.sort_values().plot(
    kind="barh"
)

plt.title(
    "15 Variáveis Mais Importantes - Random Forest"
)

plt.xlabel("Importância")
plt.show()


# ============================================================
# 25. IMPORTÂNCIA DAS VARIÁVEIS - XGBOOST
# ============================================================

importancias_xgb = pd.Series(
    modelo_xgb.feature_importances_,
    index=X_train.columns
)

importancias_xgb = importancias_xgb.sort_values(
    ascending=False
).head(15)

plt.figure(figsize=(8, 6))

importancias_xgb.sort_values().plot(
    kind="barh"
)

plt.title(
    "15 Variáveis Mais Importantes - XGBoost"
)

plt.xlabel("Importância")
plt.show()


# ============================================================
# 26. SHAP
# ============================================================

amostra_shap = X_test.sample(
    n=min(1000, len(X_test)),
    random_state=42
)

explainer = shap.TreeExplainer(
    modelo_xgb
)

shap_values = explainer.shap_values(
    amostra_shap
)


# ============================================================
# 27. SHAP - IMPORTÂNCIA GLOBAL
# ============================================================

shap.summary_plot(
    shap_values,
    amostra_shap
)


# ============================================================
# 28. SHAP - IMPORTÂNCIA EM BARRAS
# ============================================================

shap.summary_plot(
    shap_values,
    amostra_shap,
    plot_type="bar"
)


# ============================================================
# 29. SHAP - EXEMPLO INDIVIDUAL
# ============================================================

indice = 0

shap.force_plot(
    explainer.expected_value,
    shap_values[indice],
    amostra_shap.iloc[indice],
    matplotlib=True
)

plt.show()


# ============================================================
# 30. TABELA FINAL
# ============================================================

resultado_final = pd.DataFrame([

    resultado_lr,
    resultado_rf,
    resultado_xgb,
    resultado_over,
    resultado_under

])

print("\n" + "=" * 70)
print("RESULTADO FINAL DO PROJETO")
print("=" * 70)

display(
    resultado_final.sort_values(
        by="Recall",
        ascending=False
    )
)

print("\nLimiar utilizado no XGBoost:")
print(f"{melhor_limiar:.2f}")

print("\nPROJETO CONCLUÍDO!")
