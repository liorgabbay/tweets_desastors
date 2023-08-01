import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA

from preproccess_data import train_preprocess
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def pca3dim(X_train,y_train,fig):
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(X_train)
    ev = pca.singular_values_ ** 2
    x = DataFrame(np.array([ev, ev / sum(ev), pca.explained_variance_ratio_]),
              columns=["PC 1", "PC 2", "PC3"],
              index=["PCA Eigenvalues", "Explained Variance", "sklearn's Explained Variance"])

    fig1, axs = plt.subplots(nrows=1, ncols=len(x.columns), figsize=(20, 6))

    for i, col in enumerate(x.columns):
        axs[i].bar(x.index, x[col])
        axs[i].set_title(f'Principal Component {i + 1}')
        axs[i].set_ylabel('Values')
        axs[i].tick_params(axis='x', rotation=45)
        for j, val in enumerate(x[col]):
            axs[i].text(j, val, f'{val:.2f}', ha='center', va='bottom')

    plt.suptitle('Eigenvalues and Explained Variance for Each Principal Component', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("explained_pca3D.png")

    unique_labels = np.unique(y_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    ax = fig.add_subplot(121, projection='3d')
    for i, label in enumerate(unique_labels):
        mask = (y_train == label)
        ax.scatter(pca_res[mask, 0], pca_res[mask, 1], pca_res[mask, 2], color=colors[i], label=f"target {label}")

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    ax.set_title('PCA3D Visualization')
    ax.grid(True)
    ax.legend()


    # # Step 3: Apply K-means clustering and visualize clusters
    # kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
    # kmeans_result = kmeans.fit_transform(pca_res)
    #
    # # Plot cluster centroids
    # for i in range(len(unique_labels)):
    #     centroid = kmeans.cluster_centers_[i]
    #     ax.scatter(centroid[0], centroid[1], centroid[2], s=800, marker='o', edgecolors='k', facecolor='none',
    #                linewidths=2, label=f'Cluster {i + 1}')


def pca2dim(X_train, y_train,fig):
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_train)
    unique_labels = np.unique(y_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    ax2 = fig.add_subplot(122)
    for i, label in enumerate(unique_labels):
        mask = (y_train == label)
        ax2.scatter(pca_res[mask, 0], pca_res[mask, 1], color=colors[i], label=f"target {label}")

    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.set_title('PCA2D Visualization ')
    ax2.legend()


    # kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
    # kmeans_result = kmeans.fit_transform(pca_res)
    #
    # # Plot cluster centroids
    # for i in range(len(unique_labels)):
    #     centroid = kmeans.cluster_centers_[i]
    #     plt.scatter(centroid[0], centroid[1], s=300, marker='o', edgecolors='k', facecolor='none',
    #                linewidths=2, label=f'Cluster {i + 1}')



def main():
    data = pd.read_csv("S_train_data.csv")
    y_data = data['target']
    X_data = data.drop(columns=['target'])
    X_train, y_train, avg = train_preprocess(X_data, y_data)
    fig = plt.figure(figsize=(12,4))
    pca3dim(X_train,y_train,fig)
    pca2dim(X_train,y_train,fig)
    fig.savefig("PCA2D_PCA3D.png")


if __name__ == '__main__':
    main()
