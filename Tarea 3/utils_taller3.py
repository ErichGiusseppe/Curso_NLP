import numpy as np
from typing import Dict, List, Any, Iterable
from sklearn.base import BaseEstimator, TransformerMixin

IDX_NUMEROS = [0, 1, 2, 3, 7]
IDX_MOOD_PRIMARIO = 4
IDX_ETIQUETA_POLARIDAD = 6

class SenticLexiconFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, lexico):
        self.lexico = lexico
        self.vocab_moods_ = None
        self.nombres_caracteristicas = None
        self.idx_vars_incluir = None
        self._lexico_lower_ = None

    def _prepare_lexicon(self):
        if self._lexico_lower_ is None:
            self._lexico_lower_ = {str(k).lower(): v for k, v in self.lexico.items()}

    def fit(self, X, y=None):
        self._prepare_lexicon()
        conteos_mood = {}
        for bow in X:
            for tok, c in bow.items():
                entrada = self._lexico_lower_.get(str(tok).lower())
                if entrada is None:
                    continue
                try:
                    w = float(c)
                except Exception:
                    w = 0.0
                if w == 0.0:
                    continue
                mood = entrada[IDX_MOOD_PRIMARIO]
                conteos_mood[mood] = conteos_mood.get(mood, 0.0) + w

        self.vocab_moods_ = sorted(conteos_mood.keys()) if conteos_mood else ["#neutral"]

        nombres_dims = ["agradabilidad","atencion","sensibilidad","aptitud","valor_polaridad"]
        nombres_caracteristicas = [f"media_{n}" for n in nombres_dims]
        idx_vars_incluir = [0, 3, 4]
        nombres_caracteristicas += [f"var_{nombres_dims[i]}" for i in idx_vars_incluir]
        nombres_caracteristicas += [f"mood_primario_p[{m}]" for m in self.vocab_moods_]
        nombres_caracteristicas += ["polaridad_p[neg]","polaridad_p[pos]","cobertura_lexica",
                                    "conteo_pos","conteo_neg","suma_polaridad"]

        self.nombres_caracteristicas = nombres_caracteristicas
        self.idx_vars_incluir = idx_vars_incluir
        return self

    def transform(self, X):
        self._prepare_lexicon()
        filas = [self.calcular_rep(bow) for bow in X]
        if not filas:
            return np.zeros((0, len(self.nombres_caracteristicas)), dtype=float)
        return np.vstack(filas)

    def calcular_rep(self, bow):
        total_tokens = float(sum(bow.values())) if bow else 0.0
        suma_num   = np.zeros(5, dtype=float)
        suma2_num  = np.zeros(5, dtype=float)
        peso_total = 0.0
        suma_polaridad = 0.0

        conteo_mood = np.zeros(len(self.vocab_moods_), dtype=float)
        conteo_pol  = np.zeros(2, dtype=float)

        conteo_pos = 0.0
        conteo_neg = 0.0
        aciertos_lexico = 0.0

        for tok, c in bow.items():
            entrada = self._lexico_lower_.get(str(tok).lower())
            if entrada is None:
                continue
            try:
                w = float(c)
            except Exception:
                w = 0.0
            if w == 0.0:
                continue

            nums = np.array([float(entrada[i]) for i in IDX_NUMEROS], dtype=float)
            suma_num   += w * nums
            suma2_num  += w * (nums ** 2)
            peso_total += w
            suma_polaridad += w * nums[-1]

            mood = entrada[IDX_MOOD_PRIMARIO]
            if mood in self.vocab_moods_:
                idx = self.vocab_moods_.index(mood)
                conteo_mood[idx] += w

            if entrada[IDX_ETIQUETA_POLARIDAD] == "negative":
                conteo_pol[0] += w
                conteo_neg += w
            else:
                conteo_pol[1] += w
                conteo_pos += w

            aciertos_lexico += w

        if peso_total > 0:
            medias = (suma_num / peso_total)
            varianzas_all = (suma2_num / peso_total) - (medias ** 2)
        else:
            medias = np.zeros(5, dtype=float)
            varianzas_all = np.zeros(5, dtype=float)

        varianzas = varianzas_all[self.idx_vars_incluir]

        mood_p = conteo_mood / conteo_mood.sum() if conteo_mood.sum() > 0 else conteo_mood
        pol_p  = conteo_pol  / conteo_pol.sum()  if conteo_pol.sum()  > 0 else conteo_pol
        cobertura = np.array([aciertos_lexico / (total_tokens + 1e-9)], dtype=float)
        extras = np.array([conteo_pos, conteo_neg, suma_polaridad], dtype=float)

        salida = np.concatenate([medias, varianzas, mood_p, pol_p, cobertura, extras]).astype(float)
        return salida

    def get_feature_names_out(self):
        return np.array(self.nombres_caracteristicas, dtype=object)
