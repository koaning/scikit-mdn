# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "torch",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    return make_moons, mo, np, plt


@app.cell
def _(make_moons, np):
    n_samples = 1000
    X_full, _ = make_moons(n_samples=n_samples, noise=0.1)
    X = X_full[:, 0].reshape(-1, 1)
    Y = X_full[:, 1].reshape(-1, 1)
    Y += 0.1 * np.random.randn(n_samples, 1)
    return X, Y


@app.cell
def _(X, Y):
    from skmdn import MixtureDensityEstimator

    mdn_no_decay = MixtureDensityEstimator()
    mdn_no_decay.fit(X, Y)

    mdn_with_decay = MixtureDensityEstimator(weight_decay=1e-2)
    mdn_with_decay.fit(X, Y)

    models = {"No weight decay": mdn_no_decay, "weight_decay=0.01": mdn_with_decay}
    return (models,)


@app.cell
def _(mo, models):
    model_dropdown = mo.ui.dropdown(
        options=list(models.keys()), value="No weight decay", label="Model"
    )
    x_slider = mo.ui.slider(
        start=-1.5, stop=2.5, step=0.001, value=0.5, show_value=True, label="x"
    )
    [model_dropdown, x_slider]
    return model_dropdown, x_slider


@app.cell
def _(X, Y, model_dropdown, models, np, plt, x_slider):
    mdn = models[model_dropdown.value]
    x_val = x_slider.value
    pdf, ys = mdn.pdf(np.array([x_val]).reshape(1, -1))
    cdf = pdf.cumsum() / pdf.cumsum().max()
    bestmean = ys[np.argmax(cdf > 0.5)]

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].scatter(X, Y, s=3, alpha=0.3)
    axes[0].vlines(x=x_val, ymin=-1, ymax=1.5, color="orange", alpha=0.5)
    axes[0].scatter([x_val], [bestmean])
    axes[0].plot(pdf[0] + x_val, ys, color="orange")
    axes[0].set_title("interaction")

    axes[1].plot(ys, pdf[0])
    axes[1].set_title("pdf")

    axes[2].plot(ys, cdf)
    axes[2].scatter(ys[np.argmax(cdf > 0.5)], [0.5], color="orange")
    axes[2].set_title("cdf")

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The charts below also give a good impression of what is different after regularlarisation.
    """)
    return


@app.cell
def _(X, Y, models, plt):
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, (_title, _model) in zip([ax1, ax2], models.items()):
        _means, _quantiles = _model.predict(X, quantiles=[0.01, 0.1, 0.9, 0.99], resolution=100000)
        _ax.scatter(X, Y, s=3, alpha=0.3)
        _ax.scatter(X, _quantiles[:, 0], color="orange", s=3)
        _ax.scatter(X, _quantiles[:, 1], color="green", s=3)
        _ax.scatter(X, _quantiles[:, 2], color="green", s=3)
        _ax.scatter(X, _quantiles[:, 3], color="orange", s=3)
        _ax.scatter(X, _means, color="red", s=3)
        _ax.set_title(_title)

    plt.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
