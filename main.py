from flask import Flask, render_template, request

from inter import Article, Model, Prediction

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


def get_model() -> Model:
    #  TODO: Write the real implementation.
    class KKTEMP(Model):
        def predict(self, article: Article) -> Prediction:
            return Prediction(
                {
                    "real": 1.0 if "hop" in article.body else 0.0,
                    "fake": 1.0 if "hip" in article.body else 0.0,
                }
            )

    if "model" not in get_model.__dict__:
        get_model.model = KKTEMP()
    return get_model.model


@app.route("/<path:path>")
def static_file(path):
    return app.send_static_file(path)


@app.route("/")
def index():
    return render_template(
        "index.html",
        article=Article.empty_article(),
        prediction=Prediction.empty_prediction(),
    )


@app.route("/", methods=["POST"])
def predict():
    title = request.form["article-title"]
    body = request.form["article-body"]
    article = Article(title=title, body=body)

    model = get_model()
    pred = model.predict(article)

    return render_template("index.html", article=article, prediction=pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
