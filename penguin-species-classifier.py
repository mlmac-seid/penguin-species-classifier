# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns

# 3d figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# creating animations
import matplotlib.animation
from IPython.display import HTML

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

penguins_df = sns.load_dataset("penguins").dropna()

penguins_df

sns.pairplot(penguins_df, vars=penguins_df.columns[2:6], hue='species');

ac_df = penguins_df.loc[penguins_df['species'] != 'Gentoo']
ac_df = ac_df[['species', 'bill_length_mm', 'bill_depth_mm']]
ac_df
X = np.array(ac_df[['bill_length_mm', 'bill_depth_mm']])
X
y = np.array(ac_df['species'])
y
X.shape, y.shape

X = np.insert(X,obj=0,values=1,axis=1)
X.shape
X[:5,:]

sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", data=ac_df, hue="species")

adelie_X = X[y=='Adelie',:]
chinstrap_X = X[y=='Chinstrap',:]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(adelie_X[:,1], adelie_X[:,2], -1, label='adelie');
ax.scatter3D(chinstrap_X[:,1], chinstrap_X[:,2], 1, label='chinstrap');
ax.set_xlabel('bill length')
ax.set_ylabel('bill depth')
ax.set_zlabel('Type of Penguin')
plt.legend();

for i in range(len(y)):
  if(y[i]=='Adelie'):
    y[i] = -1
  else:
    y[i] = 1


w_lin = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

w_lin

px = np.linspace(30,60,20)
py = np.linspace(16,20,20)
pX,pY = np.meshgrid(px,py)
pZ = w_lin[0] + pX*w_lin[1] + pY*w_lin[2]

fig = plt.figure()
ax = plt.axes(projection='3d')

# plot hyper-plane
ax.plot_surface(pX, pY, pZ, alpha=0.4, color=colors[4]);

# plot classes
ax.scatter3D(adelie_X[:,1], adelie_X[:,2], -1, label='adelie');
ax.scatter3D(chinstrap_X[:,1], chinstrap_X[:,2], 1, label='chinstrap');

ax.set_xlabel('bill length')
ax.set_ylabel('bill depth')
ax.set_zlabel('Type of Penguin')
ax.set_zlim(-1,1)
plt.legend();

slope_lin = -(w_lin[0]/w_lin[2])/(w_lin[0]/w_lin[1])
intercept_lin = -w_lin[0]/w_lin[2]

plt.figure()
plt.scatter(X[y==-1,1],X[y==-1,2],label='setosa');
plt.scatter(X[y==1,1],X[y==1,2],label='versicolor');
plt.plot(px, px*slope_lin + intercept_lin, color=colors[4])
plt.xlabel('sepal length')
plt.ylabel('sepal width');
plt.legend();

from sklearn.linear_model import LogisticRegression
penguins2 = sns.load_dataset("penguins").dropna()
penguins2 = penguins2.loc[penguins_df['species'] != 'Gentoo']
penguins2 = penguins2[['species', 'bill_length_mm', 'bill_depth_mm']]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit_transform(penguins2['species'])

np.all(
    label_encoder.fit(penguins2['species']).transform(penguins2['species'])
    == label_encoder.fit_transform(penguins2['species'])
)

penguins2['class'] = label_encoder.fit_transform(penguins2['species'])

penguins2

sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", data=ac_df, hue="species")

Xlog = penguins2.iloc[:,1:3].values
ylog = penguins2['class'].values
X.shape, y.shape

adelie_X = penguins2.loc[penguins2["class"]==0]
adelie_X = adelie_X.iloc[:,1:3].values
chinstrap_X = penguins2.loc[penguins2["class"]==1]
chinstrap_X = chinstrap_X.iloc[:,1:3].values

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='none')
log_reg.fit(Xlog,ylog)

log_reg.coef_, log_reg.intercept_

def sig_curve(data):
  e_array = math.e*(np.ones_like(len(data)))
  return 1/(1+np.power(e_array, -(log_reg.intercept_ + (log_reg.coef_[0][0])*data[0:len(data), 0]+(log_reg.coef_[0][1]*data[0:len(data), 1]))))

sig_curve(XY_pl)

import math
# setup plot
x_pl = np.linspace(30,60,100)
y_pl = np.linspace(5,22,100)
X_pl,Y_pl = np.meshgrid(x_pl,y_pl)
XY_pl = np.vstack((X_pl.ravel(),Y_pl.ravel())).T
XY_pl.shape
Z = sig_curve(XY_pl).reshape(100,100)

# draw plot
import plotly.graph_objects as go



fig = go.Figure(data=[go.Surface(z=Z,
                                 x=x_pl,
                                 y=y_pl,
                                 opacity=0.2,
                                 colorscale='Turbo',
                                 showscale=False),
                      go.Scatter3d(z=sig_curve(adelie_X),
                                   x=adelie_X[:,0],
                                   y=adelie_X[:,1],
                                   mode='markers',
                                   marker=dict(
                                       size=4,
                                       color=colors[0],
                                       opacity=0.8),
                                   name='adelie'
                                  ),
                      go.Scatter3d(z=sig_curve(chinstrap_X),
                                   x=chinstrap_X[:,0],
                                   y=chinstrap_X[:,1],
                                   mode='markers',
                                   marker=dict(
                                       size=4,
                                       color=colors[1],
                                       opacity=0.8),
                                   name='chinstrap'
                      )])
fig.update_coloraxes(showscale=False)
fig.update_layout(autosize=False,
                  width=500,
                  height=500,
                  scene = dict(
                    xaxis_title='bill_length_mm',
                    yaxis_title='bill_depth_mm'),
                    margin=dict(l=0, r=0, b=0, t=0)
                 )

X.shape
y_hat = X.dot(w_lin)
y_hat[:5]
y_hat_class = np.sign(y_hat)
y_hat_class[:5]
np.sum(y_hat_class != y)
#number of errors for linear regression is 6

#logistic:
errors = log_reg.predict(Xlog) != ylog
sum_errors = 0
for i in range(len(errors)):
  if(errors[i] == True):
    sum_errors +=1
sum_errors
#number of errors for logistic regression is 6

penguins_df = ac_df
penguins_df

feature_names = ['bill_length_mm', 'bill_depth_mm']

X = penguins_df[feature_names].values
y = penguins_df['species'].replace({'Adelie':0,'Chinstrap':1}).values

from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 10

knn_model = KNeighborsClassifier(n_neighbors)
knn_model.fit(X, y)

print(f'Accuracy: {knn_model.score(X, y)*100:.2f}%')

# decision boundary plotting function
from matplotlib.colors import ListedColormap
def plot_decision_boundary(model, X, y, scale=1):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h=0.5
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h*scale))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    cmap = sns.palettes.color_palette('muted',as_cmap=True)
    cmap_light = ListedColormap(cmap[:2])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light,alpha=0.5);
    # Plot also the training points
    ax = sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        alpha=1.0,
        edgecolor="black",
        palette='muted'
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'KNN: K={n_neighbors} and Accuracy: {model.score(X, y)*100:.2f}%');
    plt.xlabel(feature_names[0]);
    plt.ylabel(feature_names[1]);
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Adelie','Chinsrap'])

plot_decision_boundary(knn_model,X,y)

ac2_df = sns.load_dataset("penguins").dropna()
ac2_df = ac2_df.loc[ac2_df['species'] != 'Gentoo']
feature_names = ['flipper_length_mm', 'body_mass_g']
X, y = ac2_df[feature_names].values, ac2_df['species'].replace({'Adelie':0,'Chinstrap':1}).values
n_neighbors = 10

knn_model = KNeighborsClassifier(n_neighbors)
knn_model.fit(X, y)

plot_decision_boundary(knn_model,X,y)