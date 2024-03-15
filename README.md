# Testing DFT












## lecture notes

### DFT Procedure
1. Construct $V_{ion}$ given atomic numbers and positions of ions.
2. Pick a cutoff for the plane wave basis set $\exp(i(\vec G \cdot \vec r))$
3. Pick a trial density $n(\vec r)$
4. Calculate $V_H(n)$ and $V_{XC}(n)$
5. Solve
$
H \Psi = \left( -\frac{\hbar^2 \nabla^2}{2m} + V_{ion} + V_{H} + V_{XC} \right) \Psi = E_0\Psi
$, by diagonalization of $H_{\vec G,\vec G ^\prime }$
- Calculate new $n(\vec r) = \sum_i \left| \phi_i(\vec r) \right|^2$
- Is solution self consistent?
    - Yes &rarr; Done, compute total energy
    - No &rarr; Back to step 4

### Kohn-Sham equations
$$
\left[ -\frac{\hbar^2}{2m} + V_s(\vec r) \right] \phi_i(\vec r) = \epsilon_i \phi_i(\vec r)
$$
$$
V_s = V + \int \frac{e^2 n_s(\vec r ^\prime)}{\left|\vec r - \vec r ^\prime \right|} d^3 r^\prime + V_{XC}\left[n_s(\vec r)\right]
$$
$$
n(\vec r)=\sum_i \left|\phi_i(\vec r)\right|^2
$$