import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


import itertools



class Experimentacion_modelo:
    ## Atributes
    def __init__(self, model, hiper_parms, x_train, y_train, x_test=None, k_fold=4):
        self.model = model
        self.hiperparametros = hiper_parms  # Diccionario de hiperparametros
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        # self.y_test = y_test
        self.k_fold = KFold(n_splits=k_fold, shuffle=True, random_state=42)

        # atributos para guardar resultados
        self.resultados = (
            list()
        )  # lista de resultados [acuracy, presicion, recall, f1] de cada combinacion de hiperparametros
        self.combinaciones_hiper = None  # Guardar combinaciones de hiperparametros

        # best hiperparametros
        self.best_hiper = None

    def set_combinaciones(self):
        combinaciones_hiper = []
        for tupla in itertools.product(*self.hiperparametros.values()):
            tipos_originales = [type(valor) for valor in self.hiperparametros.values()]
            nueva_tupla = [valor if isinstance(tipo, type) else [valor] for tipo, valor in zip(tipos_originales, tupla)]
            combinaciones_hiper.append(nueva_tupla)

        self.combinaciones_hiper = combinaciones_hiper
        return combinaciones_hiper

    ## Methods
    def experimentacion(self):
        # Generar combinaciones de hiperparametros a probar
        combinaciones_hiper = self.set_combinaciones()

        # Entrenar modelo y encontrar hiperparametros
        for hiper in combinaciones_hiper:
            result_temp = []
            # Splitear datos de entrenamiento en k partes (k-fold cross validation)
            for train_index, test_index in self.k_fold.split(self.x_train):
                x_train_k, y_train_k = (
                    self.x_train[train_index],
                    self.y_train[train_index],
                )
                x_test_k, y_test_k = self.x_train[test_index], self.y_train[test_index]

                # Entrenar modelo
                self.model.set_params(*hiper)
                self.model.train(x_train_k, y_train_k)

                # Evaluar modelo
                y_pred = self.model.predict(x_test_k)

                # Guardar resultados de los scores obtenidos
                result_temp.append(self.get_metrics_macro(y_test_k, y_pred))
                # la otra es usar precision_recall_curve

            # Sacar los promedios de cada score obtenido en cada fold
            result_temp = np.array(result_temp)
            result_temp = np.mean(result_temp, axis=0)
            self.resultados.append(result_temp)

        # Extraer los mejores hiperparametros
        self.best_hiper = self.extract_best_hiper()
        return {key: value for key, value in zip(self.hiperparametros.keys(), self.best_hiper[0])}

    def get_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def get_precision(self, y_true, y_pred, tipo='macro', zero_division=1):
        return precision_score(y_true, y_pred, average=tipo, zero_division=zero_division)

    def get_recall(self, y_true, y_pred, tipo='macro', zero_division=1):
        return recall_score(y_true, y_pred, average=tipo, zero_division=zero_division)

    def get_f1(self, y_true, y_pred, tipo='macro', zero_division=1):
        return f1_score(y_true, y_pred, average=tipo, zero_division=zero_division)

    def get_metrics(self, y_true, y_pred, tipo='macro', zero_division=1):
        accuracy = self.get_accuracy(y_true, y_pred)
        precision = self.get_precision(y_true, y_pred, tipo=tipo, zero_division=zero_division)
        recall = self.get_recall(y_true, y_pred, tipo=tipo, zero_division=zero_division)
        f1 = self.get_f1(y_true, y_pred, tipo=tipo, zero_division=zero_division)
        return [accuracy, precision, recall, f1]

    def get_metrics_macro(self, y_true, y_pred):
        # No tiene en cuenta las incidencias los labels
        return self.get_metrics(y_true, y_pred, tipo='macro')

    def get_metrics_weighted(self, y_true, y_pred):
        # Tiene en cuenta las incidencias los labels
        return self.get_metrics(y_true, y_pred, tipo='weighted')

    def get_metrics_x_label(self, y_true, y_pred):
        labels = np.unique(y_true)  # Obtener las etiquetas únicas en y_true
        metrics_dict = {}

    def get_metrics_x_label(self, y_true, y_pred):
        labels = list(sorted(set(y_true)))

        metrics_dict = {}
        for label in labels:
            precision = precision_score(y_true, y_pred, labels=[label], average=None, zero_division=1)
            recall = recall_score(y_true, y_pred, labels=[label], average=None, zero_division=1)
            f1 = f1_score(y_true, y_pred, labels=[label], average=None, zero_division=1)
            metrics_dict[label] = {
                'precision': precision[0],
                'recall': recall[0],
                'f1_score': f1[0]
            }
        return metrics_dict

    def extract_best_hiper(self):
        # Sacar el indice del mejor resultado
        # Sacar los hiperparametros que dieron el mejor resultado
        resultados_ordenados = sorted(enumerate(self.resultados), key=lambda x: (x[1][3],x[1][0]), reverse=True)
        # Extraer los índices de los 5 mejores
        mejores_indices = [indice for indice, _ in resultados_ordenados[:5]]
        best_hiper = [self.combinaciones_hiper[indice] for indice in mejores_indices]
        self.best_hiper = best_hiper
        return best_hiper

    # ploteo de resultados
    def get_matrix_confusion(self, model, i=0):
        X_train, X_test, Y_train, Y_test = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=42)
        model.set_params(*self.best_hiper[i])
        model.train(X_train, Y_train)

        Y_pred = model.predict(X_test)
        matriz_confusion = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_test))
        # Plotear matriz de confusión
        self.plot_matrix_confusion(matriz_confusion, Y_test, Y_pred)
        return matriz_confusion

    def plot_matrix_confusion(self, matriz, y_t, y_p):
        labels = [1, 2, 3, 4, 5, 6]
        sns.heatmap(matriz, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.title("Confusion Matrix of the example training with the best hyperparams", fontsize=16)
        plt.show()

        # PLotear Roc auc
        self.plot_multiclass_roc(y_t, y_p, n_classes=6)

        # Imprimer las metricas por cada label
        metrics = self.get_metrics_x_label(y_t, y_p)
        for label, metric in metrics.items():
            print(f"Label {label}:")
            print(f"  Precision: {metric['precision']:.2f}")
            print(f"  Recall: {metric['recall']:.2f}")
            print(f"  F1-Score: {metric['f1_score']:.2f}")
            print()

    def plot_f1_vs_accuracy(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        accuracy = resultados[:, 0]
        f1 = resultados[:, 3]

        # Plotear
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=accuracy, y=f1, s=80, color="blue", edgecolor="w", linewidth=1.5)
        plt.xlabel("Accuracy")
        plt.ylabel("F1")
        plt.title("F1 vs Accuracy")
        plt.show()

    def plot_precision_vs_recall(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        precision = resultados[:, 1]
        recall = resultados[:, 2]

        # Plotear
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=precision, y=recall, s=80, color="green", edgecolor="w", linewidth=1.5)

        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title("Precision vs Recall")
        plt.show()

    def plot_metrics(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        accuracy = resultados[:, 0]
        precision = resultados[:, 1]
        recall = resultados[:, 2]
        f1 = resultados[:, 3]

        # Crear una lista con los nombres de las combinaciones de hiperparámetros
        num_combinaciones = len(accuracy)
        nombres_combinaciones = [f"C_Hyper_{i+1}" for i in range(num_combinaciones)]

        # Crear un DataFrame para facilitar el uso de Seaborn
        df = pd.DataFrame({
            "Metric": ["Accuracy"] * num_combinaciones + ["Precision"] * num_combinaciones + ["Recall"] * num_combinaciones + ["F1"] * num_combinaciones,
            "Score": np.concatenate([accuracy, precision, recall, f1]),
            "Combinaciones de Hiperparámetros": nombres_combinaciones * 4,
        })

        # Plotear
        fig, ax = plt.subplots(figsize=(10, 8))

        # Barras
        sns.barplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metric", data=df, ax=ax, width=0.3,legend=True)
        # Dispersion
        # Gráfico de dispersión
        # sns.scatterplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metric", data=df, ax=ax, s=60, legend=False)
        # # Lines
        # sns.lineplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metric", data=df, ax=ax, linewidth=2)

        df_numeric = df.select_dtypes(include=['number'])

        # Aplicar min() solo a las columnas numéricas
        y_limit_inf = df_numeric.min().min()
        # y_limit_inf = min(df.min())

        ax.set_ylim(y_limit_inf, 1)
        ticks = np.arange(y_limit_inf, 1.01, 0.01)
        ax.set_yticks(ticks)
        ax.autoscale_view() # autosacalar ticks

        # Lineas horizontales en el graico
        ax.grid(True, axis="y", ls="-", color="gray", alpha=0.4)

        ax.set_xlabel("Combinaciones de hiperparámetros")
        ax.set_ylabel("Scores")
        ax.set_title("Scores vs Combinaciones de hiperparámetros")
        plt.xticks(rotation=75)
        plt.legend(framealpha=0.4, loc="upper right")
        plt.tight_layout()  # Ajustar automáticamente los márgenes de la figura
        plt.show()

    def plot_multiclass_roc(self, y_real, y_pred, n_classes):
        # Binarizar las etiquetas de las clases para poder calcular ROC
        y_real = label_binarize(y_real, classes=np.arange(n_classes))
        y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

        # Calcular ROC y AUC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_real[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Colores para cada clase
        colors = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple"]

        # Plotear todas las curvas ROC
        plt.figure(figsize=(10, 8))
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"Clase {i} (área = {roc_auc[i]:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Tasa de Falsos Positivos", fontsize=14)
        plt.ylabel("Tasa de Verdaderos Positivos", fontsize=14)
        plt.title("Curvas ROC Multiclase", fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generater_reporte(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        accuracy = resultados[:, 0]
        precision = resultados[:, 1]
        recall = resultados[:, 2]
        f1 = resultados[:, 3]

        # Extraer los mejores hiperparametros
        # best_hiper = self.best_hiper

        # Generar reporte
        reporte = pd.DataFrame(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Hiperparametros": self.combinaciones_hiper,
            }
        )
        return reporte

    def training_model_direct(self, hiper):
        self.model.set_params(*hiper)
        self.model.train(self.x_train, self.y_train)

    # Entrenar modelo con los mejores hiperparametros para la predicción final
    def training_model(self, i=0):
        self.model.set_params(*self.best_hiper[i])
        self.model.train(self.x_train, self.y_train)

    # Entrenar modelo sin tener que esperar

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred
