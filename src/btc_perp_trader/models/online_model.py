from river import compose, feature_extraction, linear_model, preprocessing, metrics
import pickle, pathlib, logging

MODEL_PATH = pathlib.Path(__file__).resolve().parents[2] / "models" / "online_river.pkl"
logger = logging.getLogger("OnlineModel")


class OnlineModel:
    def __init__(self):
        if MODEL_PATH.exists():
            self.model = pickle.loads(MODEL_PATH.read_bytes())
            logger.info("\ud83d\udd39 Modelo River carregado.")
        else:
            self._build_fresh()

        self.metric = metrics.ROCAUC()

    def _build_fresh(self):
        tfidf = feature_extraction.TFIDF(on="headline")
        scaler = preprocessing.StandardScaler()
        lm = linear_model.LogisticRegression()
        self.model = compose.Pipeline(tfidf | scaler | lm)
        logger.info("\ud83d\udd38 Modelo River criado do zero.")

    def predict(self, x: dict):
        p = self.model.predict_proba_one(x).get(1, 0.5)
        return p  # prob de subir; 1-p é short

    def learn(self, x: dict, y: int):
        self.model.learn_one(x, y)
        pickle.dump(self.model, MODEL_PATH.open("wb"))
        self.metric.update(y, self.predict(x))

    def roc(self):
        return self.metric.get()


ONLINE_MODEL = OnlineModel()  # singleton
