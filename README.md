**1. Propagación hacia atrás (backpropagation)**

La propagación hacia atrás es el proceso mediante el cual una red neuronal **calcula cómo ajustar sus pesos y sesgos** para **reducir el error** de su salida.  
Matemáticamente, se basa en **la regla de la cadena** del cálculo diferencial.

---

**2. El contexto básico: una red neuronal simple**

Supongamos una red con:
- Una entrada $x$,
- Un peso $w$,
- Un sesgo $b$,
- Una activación $a$,
- Una función de activación $\sigma$,
- Una salida $y$ real,
- Una función de pérdida $L(a, y)$.

El modelo es:

$$
z = w x + b
$$
$$
a = \sigma(z)
$$
$$
L = L(a, y)
$$

---

**3. Objetivo de la propagación hacia atrás**

Queremos calcular:
- $\frac{\partial L}{\partial w}$ (cómo cambia la pérdida si cambio el peso)
- $\frac{\partial L}{\partial b}$ (cómo cambia la pérdida si cambio el sesgo)

Para después **actualizar** los parámetros con, por ejemplo, **gradiente descendente**:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$
$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

donde $\eta$ es la tasa de aprendizaje.

---

**4. Aplicando la regla de la cadena**

Primero, calculamos:

- Para el peso $w$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

- Para el sesgo $b$:

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b}$$

Ahora derivamos cada parte:

- $\frac{\partial z}{\partial w} = x$ (porque $z = w x + b$),
- $\frac{\partial z}{\partial b} = 1$,
- $\frac{\partial a}{\partial z} = \sigma'(z)$ (derivada de la función de activación),
- $\frac{\partial L}{\partial a}$ depende de cómo sea la función de pérdida (por ejemplo, en MSE: $L(a, y) = \frac{1}{2}(a - y)^2$, entonces $\frac{\partial L}{\partial a} = a - y$).

---

**5. Fórmulas finales**

Así, combinando:

$$
\frac{\partial L}{\partial w} = (a - y) \cdot \sigma'(z) \cdot x
$$
$$
\frac{\partial L}{\partial b} = (a - y) \cdot \sigma'(z)
$$

**Estas son las derivadas que se usan para actualizar los parámetros.**

---
---

## Propagación hacia atrás en varias capas

$$
\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'^l(z^l)
$$

Esta fórmula **describe cómo calcular el error** en una **capa oculta $l$** de la red, **propagando** el error que ya calculamos en la capa siguiente $l+1$.

---

## Paso a paso:

### 1. $\delta^l$

$\delta^l$ es el **error en la capa $l$**.  
Formalmente:

$$
\delta^l = \frac{\partial L}{\partial z^l}
$$

Es decir, cuánto cambia la pérdida $L$ si hago un pequeño cambio en el valor lineal $z^l$ (el input a la activación en la capa $l$).

**NOTA:** No es con respecto a la activación $a^l$, sino al input **antes** de la activación.

---

### 2. $(W^{l+1})^T \delta^{l+1}$

Esto es:

- **$W^{l+1}$**: son los pesos de la capa **siguiente** ($l+1$), los que conectan de la capa actual $l$ a la siguiente $l+1$.
- **$(W^{l+1})^T$**: el peso se transpone para ir **hacia atrás** en lugar de hacia adelante (el peso originalmente lleva señales de $l$ hacia $l+1$, pero ahora queremos propagar de $l+1$ hacia $l$).
- **$\delta^{l+1}$**: el error ya calculado en la capa siguiente.

Así que:
$$
(W^{l+1})^T \delta^{l+1}
$$
propaga el error de vuelta: ¿cómo el error de la siguiente capa depende de la activación de la capa actual?

---

### 3. $\sigma'^l(z^l)$

- Es la **derivada de la función de activación** de la capa $l$, evaluada en $z^l$.
- Ajusta la magnitud del error dependiendo de **qué tan sensible** la activación fue a cambios en $z^l$.

Por ejemplo:
- Si usamos **ReLU**, $\sigma'(z^l)$ es 1 si $z^l > 0$, 0 si $z^l \leq 0$,
- Si usamos **sigmoide**, $\sigma'(z^l) = \sigma(z^l)(1 - \sigma(z^l))$.

---

### 4. $\odot$ (producto elemento a elemento)

- El símbolo $\odot$ indica el **producto Hadamard**: es **multiplicación elemento a elemento** (no es multiplicación matricial).
- Se hace componente por componente:
  - Si tienes vectores $u, v \in \mathbb{R}^n$,
  - Entonces $(u \odot v)_i = u_i v_i$.

---

## 5. Ejemplo
Una red de 3 capas:

* Capa 1: entrada $x \in \mathbb{R}^2$,
* Capa 2: oculta $a^1 \in \mathbb{R}^3$,
* Capa 3: salida $a^2 \in \mathbb{R}^1$.

Funciones de activación: ReLU en oculta, sigmoide en salida.

**Forward pass:**

$$
z^1 = W^1 x + b^1 \quad (W^1 \in \mathbb{R}^{3 \times 2})
$$

$$
a^1 = \text{ReLU}(z^1)
$$

$$
z^2 = W^2 a^1 + b^2 \quad (W^2 \in \mathbb{R}^{1 \times 3})
$$

$$
a^2 = \text{sigmoid}(z^2)
$$

**Backward pass:**

* Error en capa 2:

$$
\delta^2 = (a^2 - y) \odot \sigma'(z^2)
$$

donde $\sigma'(z^2) = a^2(1 - a^2)$ porque la derivada de sigmoide es eso.

* Gradientes de capa 2:

$$
\frac{\partial L}{\partial W^2} = \delta^2 (a^1)^T
$$

$$
\frac{\partial L}{\partial b^2} = \delta^2
$$

* Error en capa 1:

$$
\delta^1 = (W^2)^T \delta^2 \odot \text{ReLU}'(z^1)
$$

donde $\text{ReLU}'(z^1) = 1$ si $z^1 > 0$, $0$ si $z^1 \leq 0$.

* Gradientes de capa 1:

$$
\frac{\partial L}{\partial W^1} = \delta^1 (x)^T
$$

$$
\frac{\partial L}{\partial b^1} = \delta^1
$$

---

Perfecto. Vamos a complementar tu ejemplo con **valores numéricos concretos** y calcular todo paso a paso, tanto la **propagación hacia adelante** como la **propagación hacia atrás**.

---

## 🔢 Supuestos numéricos

Entrada:

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

Parámetros de la red:

##### Capa 1 (entrada a oculta):

$$
W^1 = \begin{bmatrix} 0.1 & 0.3 \\ -0.2 & 0.4 \\ 0.5 & -0.6 \end{bmatrix}, \quad b^1 = \begin{bmatrix} 0.1 \\ 0.0 \\ -0.1 \end{bmatrix}
$$

##### Capa 2 (oculta a salida):

$$
W^2 = \begin{bmatrix} 0.2 & -0.1 & 0.3 \end{bmatrix}, \quad b^2 = \begin{bmatrix} 0.05 \end{bmatrix}
$$

##### Etiqueta verdadera:

$$
y = \begin{bmatrix} 1 \end{bmatrix}
$$

---

### **Forward Pass**

##### Capa 1

$$
z^1 = W^1 x + b^1 = 
\begin{bmatrix}
0.1 & 0.3 \\
-0.2 & 0.4 \\
0.5 & -0.6
\end{bmatrix}
\begin{bmatrix}
1 \\ 2
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\ 0.0 \\ -0.1
\end{bmatrix}
$$

$$
= \begin{bmatrix}
0.1(1) + 0.3(2) \\
-0.2(1) + 0.4(2) \\
0.5(1) + (-0.6)(2)
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\ 0.0 \\ -0.1
\end{bmatrix}
$$

$$
= \begin{bmatrix}
0.1 + 0.6 + 0.1 \\
-0.2 + 0.8 + 0.0 \\
0.5 - 1.2 - 0.1
\end{bmatrix}
$$

$$
=\begin{bmatrix}
0.8 \\ 0.6 \\ -0.8
\end{bmatrix}
$$

$$
a^1 = \text{ReLU}(z^1) = \begin{bmatrix} \max(0, 0.8) \\ \max(0, 0.6) \\ \max(0, -0.8) \end{bmatrix} = \begin{bmatrix} 0.8 \\ 0.6 \\ 0.0 \end{bmatrix}
$$

---

#### Capa 2

$$
z^2 = W^2 a^1 + b^2 = \begin{bmatrix} 0.2 & -0.1 & 0.3 \end{bmatrix}
\begin{bmatrix} 0.8 \\ 0.6 \\ 0.0 \end{bmatrix} + 0.05
$$

$$
= 0.2(0.8) + (-0.1)(0.6) + 0.3(0.0) + 0.05 = 0.16 - 0.06 + 0 + 0.05 = 0.15
$$

$$
a^2 = \text{sigmoid}(z^2) = \frac{1}{1 + e^{-0.15}} \approx 0.5374
$$

---

## **Backward Pass**

#### Capa 2 (salida)

$$
\delta^2 = (a^2 - y) \cdot \sigma'(z^2)
$$

Sabemos que:

$$
a^2 = 0.5374, \quad y = 1 \Rightarrow a^2 - y = -0.4626
$$

La derivada de la sigmoide:

$$
\sigma'(z^2) = a^2(1 - a^2) = 0.5374 \cdot (1 - 0.5374) \approx 0.5374 \cdot 0.4626 \approx 0.2486
$$

Entonces:

$$
\delta^2 = -0.4626 \cdot 0.2486 \approx -0.1150
$$

#### Gradientes de capa 2:

$$
\frac{\partial L}{\partial W^2} = \delta^2 (a^1)^T = -0.1150 \cdot \begin{bmatrix} 0.8 & 0.6 & 0.0 \end{bmatrix} = \begin{bmatrix} -0.092 & -0.069 & 0.0 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^2} = \delta^2 = -0.1150
$$

---

### Capa 1

$$
\delta^1 = (W^2)^T \delta^2 \odot \text{ReLU}'(z^1)
$$

Primero:

$$
(W^2)^T = \begin{bmatrix} 0.2 \\ -0.1 \\ 0.3 \end{bmatrix}
$$

$$
(W^2)^T \delta^2 = \begin{bmatrix} 0.2 \\ -0.1 \\ 0.3 \end{bmatrix} \cdot (-0.1150) = \begin{bmatrix} -0.023 \\ 0.0115 \\ -0.0345 \end{bmatrix}
$$

Luego aplicamos la derivada de ReLU:

$$
z^1 = \begin{bmatrix} 0.8 \\ 0.6 \\ -0.8 \end{bmatrix} \Rightarrow \text{ReLU}'(z^1) = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

$$
\delta^1 = \begin{bmatrix} -0.023 \\ 0.0115 \\ -0.0345 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -0.023 \\ 0.0115 \\ 0.0 \end{bmatrix}
$$

#### Gradientes de capa 1:

$$
\frac{\partial L}{\partial W^1} = \delta^1 (x)^T = 
\begin{bmatrix}
-0.023 \\ 0.0115 \\ 0.0
\end{bmatrix}
\begin{bmatrix}
1 & 2
\end{bmatrix}
$$

$$
=\begin{bmatrix}
-0.023 & -0.046 \\
0.0115 & 0.023 \\
0 & 0
\end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^1} = \delta^1 = \begin{bmatrix} -0.023 \\ 0.0115 \\ 0.0 \end{bmatrix}
$$

---

## ✅ Resumen de Gradientes

* **Capa 2:**

  * $\frac{\partial L}{\partial W^2} = \begin{bmatrix} -0.092 & -0.069 & 0 \end{bmatrix}$
  * $\frac{\partial L}{\partial b^2} = -0.1150$

* **Capa 1:**

  * $\frac{\partial L}{\partial W^1} = \begin{bmatrix}
    -0.023 & -0.046 \\
    0.0115 & 0.023 \\
    0 & 0
    \end{bmatrix}$
  * $\frac{\partial L}{\partial b^1} = \begin{bmatrix} -0.023 \\ 0.0115 \\ 0 \end{bmatrix}$

---