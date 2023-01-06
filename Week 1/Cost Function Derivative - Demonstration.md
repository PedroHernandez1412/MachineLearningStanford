
Demonstration of the [Cost Function] derivative in respect to the parameter w:

$$\frac{\partial}{\partial{w}}J(w, b) = \frac{\partial}{\partial{w}}\frac{1}{2m}\sum_{i=1}^{m}(f_{w, b}(x^{i})-y^{i})^{2}$$

$$\frac{\partial}{\partial{w}}J(w, b) = \frac{\partial}{\partial{w}}\frac{1}{2m}\sum_{i=1}^{m}(wx^{i}+b-y^{i})^{2}$$

$$\frac{\partial}{\partial{w}}J(w, b) = \frac{1}{2m}\sum_{i=1}^{m}(wx^{i}+b-y^{i})2x^{i}$$

$$\frac{\partial}{\partial{w}}J(w, b) = \frac{1}{m}\sum_{i=1}^{m}(wx^{i}+b-y^{i})x^{i}$$


That's the reason why the [Cost Funcion] have a two dividing the number of examples in the dataset, bscause it cancels out the two that appears from computing the derivative.

$$\frac{\partial}{\partial{b}}J(w, b) = \frac{\partial}{\partial{b}}\frac{1}{2m}\sum_{i=1}^m(f_{w, b}(x^{i})-y^{i})^{2}$$

$$\frac{\partial}{\partial{b}}J(w, b) = \frac{\partial}{\partial{b}}\frac{1}{2m}\sum_{i=1}^m(wx^{i}+b-y^{i})^{2}$$

$$\frac{\partial}{\partial{b}}J(w, b) = \frac{1}{2m}\sum_{i=1}^m(wx^{i}+b-y^{i})2$$

$$\frac{\partial}{\partial{b}}J(w, b) = \frac{1}{m}\sum_{i=1}^m(wx^{i}+b-y^{i})$$





