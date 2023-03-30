#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# I zapisujemy go w dataframe:
dataset = pd.read_csv(url, names=names)



# rozdzielamy go na kolumny z cechami i wartościami wzorcowymi (etykietami klas)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# Następnie dzielimy na zbiór „uczący” i testowy (shape w cudzysłowie, bo nie ma tu uczenia):
from sklearn.model_selection import train_test_split

# yx = pd.DataFrame([y,y])
# yx[[2]]
# import numpy as np
# xyy = np.array(yx)
# xyy[:,:1]

# yx.iloc[1,:] = "test" 
# yx.iloc[0.:] = yx.iloc[1,:]
# y_int, classes = pd.factorize(yx.iloc[1,:])


y_int, classes = pd.factorize(y)
X_train, X_test, y_train, y_test, y_int_train, y_int_test = train_test_split(X, y, y_int, test_size=0.20)


# Z wykładu wiemy, że dobrze jest dane znormalizować, aby duże wartości jednej zmiennej nie dominowały liczeniu odległości Euklidesowej:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Proszę cały czas sprawdzać jakie mamy wartości w naszych zbiorach i podać to w sprawozdaniu.  Teraz możemy uruchomić klasyfikator kNN
from sklearn.neighbors import KNeighborsClassifier


def fun(k=3):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    err_list = y_test == y_pred
    err = 0
    for i in err_list:
        if i == False:
            err=err+1
    print(err / len(y_test))
    return err / len(y_test)

#%%
errors = [fun(k) for k in range(1,20)]

#%%
# plot
fig, ax = plt.subplots()
ax.plot(range(1,20), errors, linewidth=2.0)
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
plt.show()


x = X_test[0]



#%%
n_neighbors=2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay
# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    classifier.fit(X[:,:2], y)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        classifier,
        X[:,:2],
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=dataset.columns[0],
        ylabel=dataset.columns[1],
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

plt.show()

#%%
import torch
intY, classes = pd.factorize(y)
pX_train, pX_test, py_train, py_test = train_test_split(X, intY, test_size=0.20)

pX_train = torch.Tensor(pX_train)
py_train = torch.Tensor(py_train)
pX_test = torch.Tensor(pX_test)
py_test = torch.Tensor(py_test)

size1 = pX_test.size()
size2 = pX_train.size()

k = 3

#distance
d1 = pX_test.unsqueeze(0).expand(size2[0],-1,-1)
d2 = pX_train.unsqueeze(1).expand(-1,size1[0],-1)
distance = (d1 - d2).pow(2).sum(dim=2)

neig_value, neig_index = distance.topk(k, largest=False, dim=0)


neig_class = py_train[neig_index.view(-1)]
neig_class = neig_class.view(k,-1)

res_clas, inv, res_clas_count = neig_class.contiguous().unique(return_inverse=True, return_counts=True, dim=0)
# res_clas, res_clas_count = torch.unique(neig_index,return_counts=True, dim=1)
 
# index = torch.arange(0, size1[0]).unsqueeze(0).expand(k,-1)


# i = [[0, 1, 1], [2, 2, 2]]
# v =  [3,4,5]
# s = torch.sparse_coo_tensor(i, v, (2, 3))
# d = s.to_dense()



# st1 = torch.sparse_coo_tensor(neig_class, [1,1,1], (3, 30))

# i = torch.tensor([[0, 1, 1],[2, 0, 2]])

# %%
