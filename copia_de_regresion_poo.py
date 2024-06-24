from scipy.stats import expon
from scipy.stats import t
from scipy.stats import chi2
from numpy.random import randint
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
from scipy.stats import norm
import pandas as pd
import random
from sklearn.metrics import auc

class ResumenNumerico:
    def __init__(self, datos):
        self.datos = np.array(datos)

    def calculo_de_media(self):
      media = sum(self.datos)/len(self.datos)
      return media

    def calculo_de_mediana(self, datos=None):
      n = len(self.datos)
      datos_ordenados = sorted(self.datos)
      if n % 2 == 0:
        indice1 = n // 2 - 1
        indice2 = n // 2
        mediana = (datos_ordenados[indice1]+datos_ordenados[indice2])/2
      else:
        indice3 = n // 2
        mediana = datos_ordenados[indice3]

      return mediana

      return mediana

    def calculo_de_desvio_estandar(self):
        longitud = len(self.datos)
        denominador = longitud - 1
        media_desvio = sum(self.datos) / longitud
        lista_sumatoria = []

        for x in self.datos:
            resta = x - media_desvio
            cuadrado = resta ** 2
            lista_sumatoria.append(cuadrado)

        numerador = sum(lista_sumatoria)
        S = (numerador / denominador) ** 0.5

        return S

    def calculo_de_cuartiles(self):
        datos_ordenados = sorted(self.datos)
        q1 = np.percentile(datos_ordenados, 25)
        q2 = np.percentile(datos_ordenados, 50)
        q3 = np.percentile(datos_ordenados, 75)

        return [q1, q2, q3]

    def calculo_de_varianza(self):
      var = np.var(self.datos)

      return var

    def generacion_resumen_numerico(self):
        res_num = {
        'Media': self.calculo_de_media(),
        'Mediana': self.calculo_de_mediana(),
        'Desvio': self.calculo_de_desvio_estandar(),
        'Cuartiles': self.calculo_de_cuartiles(),
        "Varianza": self.calculo_de_varianza(),
        'Mínimo': min(self.datos),
        'Máximo': max(self.datos)
        }

        return res_num

    def muestra_resumen(self):
        res_num = self.generacion_resumen_numerico()
        for estad, valor in res_num.items():
          print(f"{estad}: {np.round(valor,3)}")

class ResumenGrafico:
    def __init__(self, datos):
        self.datos = np.array(datos)

    def generacion_histograma(self, h):
        val_min = min(self.datos)
        val_max = max(self.datos)
        bins = np.arange(val_min, val_max, h)
        if val_max > bins[-1]:
            bins = np.append(bins, bins[-1] + h)

        m = len(bins)
        histo = [0] * (m - 1)  # El histograma tiene m-1 bins
        for valor in self.datos:
            for i in range(len(bins) - 1):
                if valor == bins[0]:
                    histo[0] += 1
                    break
                elif bins[i] < valor <= bins[i + 1]:
                    histo[i] += 1
                    break
        for i in range(len(histo)):
            histo[i] /= (len(self.datos) * h)

        return bins, histo

    def evalua_histograma(self, h, x):
        bins, histo = self.generacion_histograma(h)

        res = [0] * len(x)
        for j in range(len(x)):
            if x[j] == min(self.datos):
                res[j] = histo[0]
            else:
                for i in range(len(bins) - 1):
                    if bins[i] < x[j] <= bins[i + 1]:
                        res[j] = histo[i]
                        break
        return res

    def kernel_gaussiano(self,x):
      valor_kernel_gaussiano = (1/(np.sqrt(2*pi)))* exp(-0.5*(x**2))
      return valor_kernel_gaussiano

    def kernel_uniforme(self,x):
      if x == -0.5:
        valor_kernel_uniforme = 1
      elif -0.5 < x <=0.5:
        valor_kernel_uniforme = 1
      else:
        valor_kernel_uniforme =  0
      return valor_kernel_uniforme

    def kernel_cuadratico(self,x):
      if x==-1:
        valor_kernel_cuadratico = 0.75 * (1-(x**2))
      elif -1<x<=1:
        valor_kernel_cuadratico = 0.75 * (1-(x**2))
      else:
        valor_kernel_cuadratico = 0
      return valor_kernel_cuadratico

    def kernel_triangular(self,x):
      if x==-1:
        valor_kernel_triangular = 1+x
      elif -1<x<=0:
        valor_kernel_triangular = 1+x
      elif x==0:
        valor_kernel_triangular = 1-x
      elif 0<x<=1:
        valor_kernel_triangular = 1-x
      else:
        valor_kernel_triangular = 0

      return valor_kernel_triangular

    def densidad_nucleo(self,h,kernel,x):
      density = np.zeros(len(x))
      data = self.datos
      for i in range(len(x)):
            contador = 0
            for j in range(len(data)):
                dato = (data[j]-x[i]) / h

                if kernel == "gaussiano":
                  contador += self.kernel_gaussiano(dato)
                elif kernel == "uniforme":
                  contador += self.kernel_uniforme(dato)
                elif kernel == "cuadratico":
                  contador += self.kernel_cuadratico(dato)
                elif kernel == "triangular":
                  contador += self.kernel_triangular(dato)

            density[i] = contador /(len(data) * h)
      return density

class GeneradoraDeDatos:
    def __init__(self, tamaño):
          self.tamaño = tamaño

    def generar_datos_dist_norm(self,media,desvio):
      tamaño = self.tamaño
      datos_normales = norm.rvs(media,desvio,tamaño)
      return datos_normales

    def pdf_norm(self,x,media,desvio):
      curva = norm.pdf(x,media,desvio)
      return curva

    def generar_datos_bs(self):
        u = np.random.uniform(size=(self.tamaño,))
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
          ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
          y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))

        return y
    def curva_bs(self,x):
      dato = 0.5* norm.pdf(x,0,1)
      sumatoria = 0
      for i in range(5):
        sumatoria = sumatoria + norm.pdf(x,(i/2)-1,0.1)
      densidad = dato + (0.1* sumatoria)
      return densidad

class Regresion:
    def __init__(self, x, y):
        # x = variables predictora/s
        # y = variable respuesta
        self.x = x
        self.y = y

    def mostrar_estadisticas(self):
        estadisticas = {
            "media": np.mean(self.y),
            "desvio": np.std(self.y),
            "minimo": np.min(self.y),
            "maximo": np.max(self.y),
            "mediana": np.median(self.y)
        }
        return estadisticas

    def evalua_histograma(self, h, x):
        bins = np.arange(min(self.y), max(self.y) + h, h)
        cant_x_intervalo = [0] * (len(bins) - 1)

        for dato in self.y:
          for i in range(len(bins) - 1) :
            if dato == bins[0]:
              cant_x_intervalo[0] += 1
              break
            elif bins[i] < dato <= bins[i+1]:
              cant_x_intervalo[i] += 1
              break

        histo = np.array(cant_x_intervalo)
        hist = histo / (len(self.y) * h)

        res = [0] * len(x)
        for j in range(len(x)):
          for i in range(len(bins) - 1):
            if x[j] == bins[0]:
              res[j] = hist[0]
            if bins[i] < x[j] <= bins[i+1]:
              res[j] = hist[i]

        return res

    def ajustar_modelo(self):
        X = sm.add_constant(self.x)  # Agregar constante
        modelo = sm.OLS(self.y, X)  # Ajustar el modelo OLS
        modelo_ajust = modelo.fit()
        return modelo_ajust

    def datos(self):
        modelo = self.ajustar_modelo()

        print("Coeficientes del modelo:")
        for i in range(len(modelo.params)):
            coef = modelo.params[i]
            print("Beta", i, ":", coef)

        print("Valores de t observados:")
        for i in range(len(modelo.params)):
            coef = modelo.params[i]
            t_obs = coef / modelo.bse[i]
            print("t observado para Beta", i, ":", t_obs)

        std_errors = modelo.bse

        print("Errores estándar de los coeficientes:")
        for i in range(len(std_errors)):
            se = std_errors[i]
            print("SE(beta", i, ".est):", se)

        p_values = modelo.pvalues

        print("Valores p:")
        for i in range(len(p_values)):
            p_value = p_values[i]
            print("p-valor para Beta", i, ":", p_value)

        IC = modelo.conf_int()

        print('Intervalos de confianza para los coeficientes:')
        for i in range(len(IC)):
            intervalo_beta = IC.iloc[i].values
            print('Intervalo de confianza para beta', i, ':', intervalo_beta)

class RegresionLinealSimple(Regresion):
    def __init__(self, x, y):    #Aca, unicamente pongo dentro de los parentesis mi "x" e "y"
        super().__init__(x, y)

    def predecir_e_intervalos(self, new_x):
        modelo = self.ajustar_modelo()
        X_new = sm.add_constant(np.array([[1, new_x]]))
        prediccion = modelo.get_prediction(X_new)
        int_conf = prediccion.conf_int()
        int_pred = prediccion.conf_int(obs=True)
        predic = np.mean(int_conf)
        return {"int_conf": int_conf, "int_pred": int_pred, "prediccion": predic}

    def graficar_recta_ajustada(self):
        modelo = self.ajustar_modelo()
        yi = modelo.params[0] + modelo.params[1] * self.x
        plt.scatter(self.x, self.y);
        plt.plot(self.x, yi);

    def coeficiente_de_correlacion(self):
        correlacion = np.corrcoef(self.y,self.x)
        return correlacion[1:2]

    def minimos_cuadrados(self):
      pass

    def residuos(self):
        modelo = self.ajustar_modelo()
        residuos = self.y - (modelo.params[0] + modelo.params[1] * self.x)
        y_i = modelo.params[0] + self.x * modelo.params[1]

        plt.figure()
        plt.scatter(y_i, residuos)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Valores ajustados')
        plt.ylabel('Residuos')
        plt.title('Gráfico de Dispersión de Residuos vs Valores Ajustados')
        plt.show()

        plt.figure()
        cuantiles_muestra = (residuos - np.mean(residuos)) / np.std(residuos)
        cuantiles_muestra = np.sort(cuantiles_muestra)

        cuantiles_teoricos = []
        for num in range(1, len(cuantiles_muestra) + 1):
            p = num / (len(cuantiles_muestra) + 1)
            q = norm.ppf(p)
            cuantiles_teoricos.append(q)

        plt.scatter(cuantiles_muestra, cuantiles_teoricos, color='blue', marker='o')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, linewidth=3.5, color="red")
        plt.xlabel('Cuantiles Muestra')
        plt.ylabel('Cuantiles Teóricos')
        plt.title('QQ Plot')
        plt.show()

        suma = sum((residuos ** 2))
        longitud_n = len(residuos)
        fraccion = 1 / (longitud_n - 2)
        desvio = fraccion * suma
        print("La varianza de los residuos es", desvio)

      #RESIDUO PUNTUAL

class RegresionLinealMultiple(Regresion):
    def __init__(self, x, y):
        super().__init__(x, y)

    def graficar_rectas_ajustadas(self):
        modelo = self.ajustar_modelo()
        num_vars = self.x.shape[1]

        cols = 3
        rows = (num_vars // cols) + int(num_vars % cols > 0)

        fig, ax = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        ax = ax.flatten()

        variables = self.x.columns
        for i, var in enumerate(variables):
            ax[i].scatter(self.x[var], self.y, label='Datos observados')
            y_i = modelo.params[0] + modelo.params[i+1] * self.x[var]
            ax[i].plot(self.x[var], y_i, color='red', label='Recta ajustada')
            ax[i].set_xlabel(var)
            ax[i].set_ylabel('stack.loss')
            ax[i].set_title(f'{var} vs stack.loss')
            ax[i].legend()


        for j in range(i+1, rows*cols):
            fig.delaxes(ax[j])

        plt.tight_layout()
        plt.show()

    def predecir(self, new_x):
        modelo = self.ajustar_modelo()
        new_x = list(new_x)
        new_x.insert(0, 1)
        X_new = np.array([new_x])
        prediccion = modelo.get_prediction(X_new)
        int_conf = prediccion.conf_int()
        int_pred = prediccion.conf_int(obs=True)
        predic = prediccion.predicted_mean[0]
        return {"int_conf": int_conf, "int_pred": int_pred, "prediccion": predic}

    def intervalos_de_confianza(self,alpha):
        modelo = self.ajustar_modelo()
        intervalo_confianza = modelo.conf_int(alpha)
        print("Intervalos de confianza para las variables predictoras:")
        print(intervalo_confianza.to_string(header=True, index=True))

    def r_cuadrado(self):
      modelo = self.ajustar_modelo()
      r_squared_2 = modelo.rsquared
      adjusted_r_squared = modelo.rsquared_adj
      print("El coeficiente de determinacion (R-cuardrado) es",r_squared_2)
      print("El coeficiente de deterinacion ajustado (R-cuadrado adj) es",adjusted_r_squared)

    def coeficiente_de_correlacion(self):
      x = self.x
      y = self.y

      correlaciones = x.apply(lambda col: np.corrcoef(col, y)[0, 1])
      print("Coeficientes de correlación:")
      print(correlaciones)

class RegresionLogistica(Regresion):

  def ajustar_modelo(self):
    X = sm.add_constant(self.x)
    modelo = sm.Logit(self.y, X)
    self.resultado = modelo.fit()
    return self.resultado

  def prediccion_con_umbral(self,umbral,x_test,resultado):   # en "resultado" va la funcion de arriba rl.ajustar_modelo()
    X_test=sm.add_constant(x_test)                           # en "x_test" van los datos de prueba o sea datos_test
    prediccion = resultado.predict(X_test)
    y_pred = 1*(prediccion >= umbral)
    return y_pred

  def matriz_de_confusion(self,y_pred,y_test):

    tablix = np.column_stack((y_pred, y_test))
    tablox = pd.DataFrame(tablix, columns=['y_pred', 'y_test'])

    default_yes = tablox['y_test'] == 1
    a = sum(default_yes[tablox['y_pred'] == 1])  # Verdaderos positivos
    c = sum(default_yes[tablox['y_pred'] == 0])  # Falsos negativos

    default_no = tablox['y_test'] == 0
    b = sum(default_no[tablox['y_pred'] == 1])  # Falsos positivos
    d = sum(default_no[tablox['y_pred'] == 0])  # Verdaderos negativos


    matriz_confusion=[[a,b],[c,d]]
    error_total_mala_clasificacion=(c+b)/(a+b+c+d)

    especificidad=d/(d+b)
    sensibilidad=a/(a+c)

    print("Mi especificidad es de", especificidad)
    print("Mi sensibilidad es de", sensibilidad)
    print("Mi error de mala clasificacion es de", error_total_mala_clasificacion)
    print("Mi matriz de confusion es:",matriz_confusion)

  def curva_roc(self,y_test,x_test):
        grilla_p = np.linspace(0, 1, 100)
        sensibilidad = []
        especificidad = []

        # Asegurarse de ajustar el modelo una vez antes del bucle
        self.ajustar_modelo()

        for p in grilla_p:
            y_pred = self.prediccion_con_umbral(p, x_test,rl.ajustar_modelo())
            tp = sum((y_test == 1) & (y_pred == 1))
            fn = sum((y_test == 1) & (y_pred == 0))
            tn = sum((y_test == 0) & (y_pred == 0))
            fp = sum((y_test == 0) & (y_pred == 1))

            sensibilidad.append(tp / (tp + fn))
            especificidad.append(tn / (tn + fp))

        sensibilidad = np.array(sensibilidad)
        especificidad = np.array(especificidad)
        especificidad_invertida = 1 - especificidad

        # Curva ROC
        plt.figure(figsize=(10, 5))

        # Gráfico de la curva ROC
        plt.subplot(1, 2, 1)
        plt.plot(especificidad_invertida, sensibilidad, label='ROC Curve')
        plt.xlabel('1 - Especificidad')
        plt.ylabel('Sensibilidad')
        plt.title('Curva ROC')
        plt.legend()

        # Gráfico de especificidad contra sensibilidad
        plt.subplot(1, 2, 2)
        plt.plot(especificidad, sensibilidad, label='Especificidad vs Sensibilidad')
        plt.xlabel('Especificidad')
        plt.ylabel('Sensibilidad')
        plt.title('Especificidad vs Sensibilidad')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Calcular el AUC
        roc_auc = auc(especificidad_invertida, sensibilidad)
        print('roc_auc: ', roc_auc)
        if 0.90 <= roc_auc <= 1:
            print('roc_auc clasificado como excelente')
        elif 0.80 <= roc_auc < 0.90:
            print('roc_auc clasificado como bueno')
        elif 0.70 <= roc_auc < 0.80:
            print('roc_auc clasificado como regular')
        elif 0.60 <= roc_auc < 0.70:
            print('roc_auc clasificado como pobre')
        else:
            print('roc_auc clasificado como fallido')

        # Encontrar el punto óptimo
        P_optimo = max(sensibilidad + (especificidad - 1))
        return print("El punto optimo es",P_optimo)
