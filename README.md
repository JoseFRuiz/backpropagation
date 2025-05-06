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

