from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent)


def display(value):
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)



print("\nЗагрузка данных")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

df = pd.read_csv("data/Mall_Customers.csv")

if df.empty:
    raise ValueError("Датасет пустой")

print("Количество объектов:", df.shape[0])
print("Количество столбцов:", df.shape[1])
print("Типы столбцов:")
print(df.dtypes.to_string())
print("Целевой переменной с метками классов в датасете нет")

print("Размерность:", df.shape)
display(df.head())
display(df.dtypes)


print("\nПропуски и дубликаты")

print("Пропущенные значения:")
display(df.isnull().sum())

print("Количество дубликатов:", df.duplicated().sum())


print("\nОписательная статистика")

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

if X.isna().any().any():
    raise ValueError("В признаках обнаружены пропуски. Обработайте их перед кластеризацией")

print("Пропущенные значения отсутствуют: обработка не требуется")
print("CustomerID исключён как идентификатор, Gender не включён в числовую модель")

display(X.describe().T.round(2))


print("\nГистограммы")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, feature in enumerate(features):
    sns.histplot(
        data=X,
        x=feature,
        bins=20,
        kde=True,
        color="steelblue",
        ax=axes[i]
    )
    axes[i].set_title(feature)

plt.tight_layout()
plt.savefig("figures/02_histograms.png", dpi=300)
plt.show()


print("\nВыбросы")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, feature in enumerate(features):
    sns.boxplot(y=X[feature], ax=axes[i])
    axes[i].set_title(feature)

plt.tight_layout()
plt.savefig("figures/02_boxplots.png", dpi=300)
plt.show()

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (
    (X < Q1 - 1.5 * IQR) |
    (X > Q3 + 1.5 * IQR)
)

display(outlier_mask.sum().to_frame("Количество выбросов"))
print("Объектов с потенциальными выбросами:", outlier_mask.any(axis=1).sum())
print("Потенциальные выбросы сохранены: исходные наблюдения не удаляются")


print("\nИнформативность признаков")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

def feature_score(columns, k=5):
    data = StandardScaler().fit_transform(X[columns])

    labels = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    ).fit_predict(data)

    return silhouette_score(data, labels)

baseline = feature_score(features)

importance_results = []

for feature in features:
    remaining = [f for f in features if f != feature]
    score = feature_score(remaining)

    importance_results.append({
        "Признак": feature,
        "Silhouette без признака": score,
        "Изменение качества": baseline - score
    })

importance_df = pd.DataFrame(importance_results)
importance_df = importance_df.sort_values(
    "Изменение качества",
    ascending=False
)

display(importance_df.round(4))

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance_df,
    x="Изменение качества",
    y="Признак",
    color="steelblue"
)

plt.axvline(0, color="black", linestyle="--")
plt.title("Влияние исключения признаков на качество кластеризации")
plt.tight_layout()
plt.savefig("figures/02_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nСтандартизация")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features,
    index=X.index
)

scaling_stats = pd.DataFrame({
    "Признак": features,
    "Среднее до": X.mean().values,
    "Ст. отклонение до": X.std(ddof=0).values,
    "Среднее после": X_scaled_df.mean().values,
    "Ст. отклонение после": X_scaled_df.std(ddof=0).values
})

numeric_cols = scaling_stats.select_dtypes(include="number").columns

scaling_stats[numeric_cols] = scaling_stats[numeric_cols].mask(
    scaling_stats[numeric_cols].abs() < 1e-10,
    0
)

display(scaling_stats.round(4))


print("\nВизуализация стандартизации")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=X, ax=axes[0])
axes[0].set_title("До стандартизации")
axes[0].tick_params(axis="x", rotation=20)

sns.boxplot(data=X_scaled_df, ax=axes[1])
axes[1].set_title("После стандартизации")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("figures/03_scaling.png", dpi=300)
plt.show()


print("\nМатрица рассеивания")

pair_grid = sns.pairplot(
    X_scaled_df,
    diag_kind="hist",
    plot_kws={"alpha": 0.7, "s": 40},
    diag_kws={"bins": 20, "color": "steelblue"}
)

pair_grid.fig.suptitle(
    "Матрица диаграмм рассеивания",
    y=1.02
)

pair_grid.fig.savefig("figures/04_pairplot.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nМетод локтя")

k_values = range(2, 11)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Количество кластеров k")
plt.ylabel("Inertia")
plt.title("Метод локтя")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.savefig("figures/04_elbow.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nСилуэтный анализ")

from sklearn.metrics import silhouette_score

silhouette_results = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    silhouette_results.append(
        silhouette_score(X_scaled, labels)
    )

k_results = pd.DataFrame({
    "k": list(k_values),
    "Inertia": inertias,
    "Silhouette Score": silhouette_results
})

display(k_results.round(4))

plt.figure(figsize=(9, 5))
plt.plot(
    list(k_values),
    silhouette_results,
    marker="o"
)
plt.xlabel("Количество кластеров k")
plt.ylabel("Silhouette Score")
plt.title("Силуэтный анализ K-means")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.savefig("figures/04_silhouette.png", dpi=300, bbox_inches="tight")
plt.show()

best_k = int(
    k_results.loc[
        k_results["Silhouette Score"].idxmax(),
        "k"
    ]
)

print("Оптимальное k по Silhouette Score:", best_k)


print("\nДендрограмма")

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X_scaled, method="ward", metric="euclidean")

plt.figure(figsize=(14, 6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True
)

plt.title("Дендрограмма метода Уорда")
plt.xlabel("Группы объектов")
plt.ylabel("Расстояние объединения")
plt.tight_layout()
plt.savefig("figures/04_dendrogram.png", dpi=300, bbox_inches="tight")
plt.show()

hierarchical_scores = []

for k in k_values:
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        metric="euclidean"
    )

    labels = model.fit_predict(X_scaled)

    hierarchical_scores.append(
        silhouette_score(X_scaled, labels)
    )

hierarchical_results = pd.DataFrame({
    "k": list(k_values),
    "Silhouette Score": hierarchical_scores
})

display(hierarchical_results.round(4))

best_k_hierarchical = int(
    hierarchical_results.loc[
        hierarchical_results["Silhouette Score"].idxmax(),
        "k"
    ]
)

print("Оптимальное k для иерархического метода:", best_k_hierarchical)

n_objects = len(X_scaled)
lower_height = Z[n_objects - best_k_hierarchical - 1, 2]
upper_height = Z[n_objects - best_k_hierarchical, 2]
cut_height = (lower_height + upper_height) / 2

plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.axhline(cut_height, color="red", linestyle="--", label=f"Разрез для k={best_k_hierarchical}")
plt.title("Дендрограмма и уровень разреза")
plt.xlabel("Группы объектов")
plt.ylabel("Расстояние объединения")
plt.legend()
plt.tight_layout()
plt.savefig("figures/04_dendrogram_cut.png", dpi=300, bbox_inches="tight")
plt.show()



print("\nКластеризация")

kmeans = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init=10,
    random_state=42
)

hierarchical = AgglomerativeClustering(
    n_clusters=best_k_hierarchical,
    linkage="ward",
    metric="euclidean"
)

labels_kmeans = kmeans.fit_predict(X_scaled)
labels_hierarchical = hierarchical.fit_predict(X_scaled)

results_df = df.copy()

results_df["KMeans_Cluster"] = labels_kmeans
results_df["Hierarchical_Cluster"] = labels_hierarchical

print("Количество кластеров K-means:", best_k)
print("Количество кластеров иерархического метода:", best_k_hierarchical)
print("Фактическое количество кластеров K-means:", len(np.unique(labels_kmeans)))
print("Фактическое количество кластеров иерархического метода:", len(np.unique(labels_hierarchical)))
print("Размерность массива меток:", labels_kmeans.shape)
print("Итоговая Inertia K-means:", round(kmeans.inertia_, 4))
print("Центры K-means в стандартизированном пространстве:")
display(pd.DataFrame(kmeans.cluster_centers_, columns=features).round(4))

display(
    results_df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
    .to_frame("K-means")
)

display(
    results_df["Hierarchical_Cluster"]
    .value_counts()
    .sort_index()
    .to_frame("Иерархический")
)


print("\nМетрики качества")

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist, pdist

def calculate_metrics(data, labels):
    labels = np.asarray(labels)
    data = np.asarray(data)
    valid = labels != -1
    data = data[valid]
    labels = labels[valid]
    cluster_ids = np.unique(labels)
    if not 2 <= len(cluster_ids) < len(data):
        print("Метрики не рассчитываются: требуется от 2 до n-1 кластеров без шума")
        return {
            "Silhouette Score": np.nan,
            "Calinski-Harabasz": np.nan,
            "Davies-Bouldin": np.nan,
            "Внутрикластерное расстояние": np.nan,
            "Расстояние между центрами": np.nan
        }

    centers = np.array([
        data[labels == cluster].mean(axis=0)
        for cluster in cluster_ids
    ])

    center_index = np.searchsorted(cluster_ids, labels)

    distances = cdist(data, centers)

    within = distances[
        np.arange(len(data)),
        center_index
    ].mean()

    between = pdist(centers).mean()

    return {
        "Silhouette Score": silhouette_score(data, labels),
        "Calinski-Harabasz": calinski_harabasz_score(data, labels),
        "Davies-Bouldin": davies_bouldin_score(data, labels),
        "Внутрикластерное расстояние": within,
        "Расстояние между центрами": between
    }

metrics_df = pd.DataFrame({
    "K-means": calculate_metrics(X_scaled, labels_kmeans),
    "Иерархический": calculate_metrics(
        X_scaled,
        labels_hierarchical
    )
}).T

display(metrics_df.round(4))


print("\nВлияние параметров")

parameter_results = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    parameter_results.append({
        "k": k,
        "Silhouette": silhouette_score(X_scaled, labels),
        "Calinski-Harabasz": calinski_harabasz_score(
            X_scaled, labels
        )
    })

parameters_df = pd.DataFrame(parameter_results)

display(parameters_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.lineplot(
    data=parameters_df,
    x="k",
    y="Silhouette",
    marker="o",
    ax=axes[0]
)

sns.lineplot(
    data=parameters_df,
    x="k",
    y="Calinski-Harabasz",
    marker="o",
    ax=axes[1]
)

axes[0].set_title("Silhouette Score")
axes[1].set_title("Calinski-Harabasz Index")

plt.tight_layout()
plt.savefig("figures/04_parameter_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nУстойчивость")

from sklearn.metrics import adjusted_rand_score

kmeans_stability = []

for seed in range(20):
    labels = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=10,
        random_state=seed
    ).fit_predict(X_scaled)

    kmeans_stability.append(
        adjusted_rand_score(labels_kmeans, labels)
    )

rng = np.random.default_rng(42)

hierarchical_stability = []

for i in range(20):
    indices = rng.choice(
        len(X_scaled),
        size=int(len(X_scaled) * 0.8),
        replace=False
    )

    labels = AgglomerativeClustering(
        n_clusters=best_k_hierarchical,
        linkage="ward",
        metric="euclidean"
    ).fit_predict(X_scaled[indices])

    hierarchical_stability.append(
        adjusted_rand_score(
            labels_hierarchical[indices],
            labels
        )
    )

stability_df = pd.DataFrame({
    "Метод": ["K-means", "Иерархический"],
    "Среднее ARI": [
        np.mean(kmeans_stability),
        np.mean(hierarchical_stability)
    ],
    "Стандартное отклонение ARI": [
        np.std(kmeans_stability),
        np.std(hierarchical_stability)
    ],
    "Минимальное ARI": [
        np.min(kmeans_stability),
        np.min(hierarchical_stability)
    ]
})

display(stability_df.round(4))


print("\nPCA")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, labels, title in zip(
    axes,
    [labels_kmeans, labels_hierarchical],
    ["K-means", "Иерархическая кластеризация"]
):
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="tab10",
        s=65,
        ax=ax
    )

    centers_3d = np.array([
        X_scaled[labels == cluster].mean(axis=0)
        for cluster in np.unique(labels)
    ])
    centers_pca = pca.transform(centers_3d)
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        marker="X",
        s=160,
        edgecolors="white",
        label="Центры"
    )
    ax.set_title(title)
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")

plt.tight_layout()
plt.savefig("figures/05_pca_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

print(
    "Объяснённая дисперсия:",
    round(pca.explained_variance_ratio_.sum(), 4)
)


print("\nПрофили кластеров")

profile_kmeans = results_df.groupby(
    "KMeans_Cluster"
)[features].agg(["mean", "count"])

profile_hierarchical = results_df.groupby(
    "Hierarchical_Cluster"
)[features].agg(["mean", "count"])

print("K-means:")
display(profile_kmeans.round(2))

print("Иерархическая кластеризация:")
display(profile_hierarchical.round(2))

print("Центры K-means в исходном масштабе:")
display(
    pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    ).round(2)
)


print("\nСогласованность методов")

ari_methods = adjusted_rand_score(
    labels_kmeans,
    labels_hierarchical
)

print("ARI между методами:", round(ari_methods, 4))

display(
    pd.crosstab(
        results_df["KMeans_Cluster"],
        results_df["Hierarchical_Cluster"]
    )
)



X_scaled_df.to_csv("results/scaled_features.csv", index=False, encoding="utf-8-sig")
scaling_stats.to_csv("results/scaling_statistics.csv", index=False, encoding="utf-8-sig")
parameters_df.to_csv("results/parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
results_df.to_csv("results/customer_clusters.csv", index=False, encoding="utf-8-sig")
metrics_df.to_csv("results/clustering_metrics.csv", encoding="utf-8-sig")
importance_df.to_csv("results/feature_importance.csv", index=False, encoding="utf-8-sig")
k_results.to_csv("results/kmeans_k_selection.csv", index=False, encoding="utf-8-sig")
hierarchical_results.to_csv("results/hierarchical_k_selection.csv", index=False, encoding="utf-8-sig")
stability_df.to_csv("results/clustering_stability.csv", index=False, encoding="utf-8-sig")
print("\nВыполнение завершено. Результаты сохранены в figures/ и results/.")
