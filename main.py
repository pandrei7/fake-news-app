from threading import Lock, Thread

from flask import Flask, render_template, request

from models import Article, BertBasedModel, Model, Prediction

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


model_mutex = Lock()


def get_model() -> Model:
    def instantiate_model() -> Model:
        model = BertBasedModel()
        return model

    if "model" not in get_model.__dict__:
        with model_mutex:
            if "model" not in get_model.__dict__:
                get_model.model = instantiate_model()
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
    # Loading the model can take a long time. Pre-load it in a separate thread.
    model_preloader = Thread(target=get_model)
    model_preloader.start()

    app.run(host="0.0.0.0", port=8080)
    model_preloader.join()
