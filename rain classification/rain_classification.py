import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_data_with_decision_boundary(clf, ax=None, support_vectors=True):

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    # x_min, x_max, y_min, y_max = -3, 3, -3, 3
    # ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    # ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(f" Decision boundaries of SV classifier")

    # if ax is None:
    #     plt.show()
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv('LHAT.csv')
    df = df.rename(columns={
        'Intensity (mm/hour)_occurrences': 'intensity',
        'Duration (hours)_occurrences': 'duration',
        'Cumulative_RF (mm)_occurrences': 'cumrf',
        'Intensity (mm/hour)_non_occurrences': 'intensity_non',
        'Duration (hours)_non_occurrences': 'duration_non',
    })
    df = df[df['intensity'].notna()].reset_index(drop=True)
    df = df[df['intensity_non'].notna()].reset_index(drop=True)

    df['logintensity'] = np.log(df['intensity'])
    df['logduration'] = np.log(df['duration'])
    df['occurence'] = True

    df2 = df.copy()
    df2['intensity'] = df['intensity_non']
    df2['duration'] = df['duration_non']
    df2['logintensity'] = np.log(df2['intensity'])
    df2['logduration'] = np.log(df2['duration'])
    df['occurence'] = False

    df = pd.concat((df, df2), axis=0).reset_index(drop=True)
    df = df[[col for col in df.columns if col not in ['intensity_non', 'duration_non']]]


    # sns.scatterplot(data=df, x='logintensity', y='logduration', hue='occurence')
    # plt.show()

    X = df[['logintensity', 'logduration']].values
    y = df['occurence'].values

    # X = df[['Slope1', 'Road_Prox1', 'River_prox', 'Lithology1', 'Landcover1', 'Aspect1', 'logintensity', 'logduration']].values
    # idx = np.where(np.sum(np.isnan(X), axis=1) == 0)[0]
    # X = X[idx]
    # y = df['occurence'].values[idx]

    p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]}
    svc = svm.SVC(kernel="rbf")

    NUM_TRIALS = 10
    nested_scores = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        # Choose cross-validation techniques for the inner and outer loops, independently of the dataset.
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=svc, param_grid=p_grid, cv=inner_cv)
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    clf.fit(X, y)
    clf.best_params_
    clf.score(X, y)

    plot_training_data_with_decision_boundary(clf, support_vectors=False)
