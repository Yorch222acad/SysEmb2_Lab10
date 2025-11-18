import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

clients = pd.DataFrame(
    {
        "saldo": [
            750, 1245, 230, 533, 490, 1000, 190, 900, 600, 50,
            1100, 930, 450, 330, 750
        ],
        "transacciones": [
            3, 1, 4, 3, 2, 1, 0, 3, 2, 1,
            0, 4, 3, 2, 0
        ],
    }
)

# Cambiado a StandardScaler
scaler = StandardScaler().fit(clients.values)
clients = pd.DataFrame(
    scaler.transform(clients.values),
    columns=["saldo", "transacciones"]
)

print(clients)

kmeans = KMeans(n_clusters=3).fit(clients.values)
print(kmeans.labels_)

clients["cluster"] = kmeans.labels_
print(kmeans.cluster_centers_, kmeans.inertia_)

plt.figure(figsize=(6, 5), dpi=100)
colors = ["red", "orange", "blue", "black", "purple", "pink", "brown"]

print(clients)

for cluster in range(kmeans.n_clusters):
    plt.scatter(
        clients[clients["cluster"] == cluster]["saldo"],
        clients[clients["cluster"] == cluster]["transacciones"],
        marker="x", s=180, color=colors[cluster], alpha=0.5
    )
    plt.scatter(
        kmeans.cluster_centers_[cluster][0],
        kmeans.cluster_centers_[cluster][1],
        marker="o", s=280, color=colors[cluster]
    )

plt.title("Clients")
plt.xlabel("Client earnings")
plt.ylabel("Client transactions")
plt.text(1.1, 0.5, "K={}".format(kmeans.n_clusters))
plt.text(1.1, 0.3, "Inertia={}".format(round(kmeans.inertia_, 2)))
plt.show()
