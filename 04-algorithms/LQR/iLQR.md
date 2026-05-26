# iLQR Derivative and Cost-to-Go Function

For a nonlinear discrete-time system:

\[
x_{t+1} = f(x_t, u_t)
\]

where:

- \(x_t\) is the state at time \(t\)
- \(u_t\) is the control input at time \(t\)
- \(f(x_t,u_t)\) describes the nonlinear system dynamics

The goal of iLQR is to find an optimal control sequence that minimizes a nonlinear trajectory cost.

---

## 1. iLQR Objective Function

The total cost from time \(t\) to the final time \(T\) is:

\[
J_t =
\sum_{k=t}^{T-1}
\ell(x_k,u_k)
+
\ell_f(x_T)
\]

where:

- \(\ell(x_k,u_k)\) is the running cost at each time step
- \(\ell_f(x_T)\) is the terminal cost
- \(x_{k+1}=f(x_k,u_k)\) follows nonlinear dynamics

The objective is:

\[
\min_{u_t,u_{t+1},\dots,u_{T-1}} J_t
\]

---

## 2. Cost-to-Go Function

Define the cost-to-go function:

\[
V_t(x_t)
\]

This means the minimum future cost starting from state \(x_t\) at time \(t\).

Using dynamic programming:

\[
V_t(x_t)
=
\min_{u_t}
\left[
\ell(x_t,u_t)
+
V_{t+1}(x_{t+1})
\right]
\]

Since:

\[
x_{t+1}=f(x_t,u_t)
\]

we write:

\[
V_t(x_t)
=
\min_{u_t}
\left[
\ell(x_t,u_t)
+
V_{t+1}(f(x_t,u_t))
\right]
\]

---

## 3. Local Approximation Around a Nominal Trajectory

Because the system is nonlinear, iLQR does not solve the full problem directly.

Instead, it starts with a nominal trajectory:

\[
\bar{x}_t,\bar{u}_t
\]

and defines small deviations:

\[
\delta x_t = x_t - \bar{x}_t
\]

\[
\delta u_t = u_t - \bar{u}_t
\]

The nonlinear dynamics are linearized around the nominal trajectory:

\[
\delta x_{t+1}
\approx
f_x \delta x_t + f_u \delta u_t
\]

where:

\[
f_x =
\frac{\partial f}{\partial x}
\]

\[
f_u =
\frac{\partial f}{\partial u}
\]

Both derivatives are evaluated at the nominal point:

\[
(\bar{x}_t,\bar{u}_t)
\]

---

## 4. Quadratic Approximation of the Cost

The running cost is approximated using a second-order Taylor expansion:

\[
\ell(x_t,u_t)
\approx
\ell
+
\ell_x^T \delta x_t
+
\ell_u^T \delta u_t
+
\frac{1}{2}\delta x_t^T \ell_{xx}\delta x_t
+
\delta u_t^T \ell_{ux}\delta x_t
+
\frac{1}{2}\delta u_t^T \ell_{uu}\delta u_t
\]

where:

\[
\ell_x = \frac{\partial \ell}{\partial x}
\]

\[
\ell_u = \frac{\partial \ell}{\partial u}
\]

\[
\ell_{xx} = \frac{\partial^2 \ell}{\partial x^2}
\]

\[
\ell_{uu} = \frac{\partial^2 \ell}{\partial u^2}
\]

\[
\ell_{ux} = \frac{\partial^2 \ell}{\partial u \partial x}
\]

---

## 5. Local Q-Function

iLQR defines a local Q-function:

\[
Q(\delta x_t,\delta u_t)
=
\ell(x_t,u_t)
+
V_{t+1}(x_{t+1})
\]

Using the local linear dynamics and quadratic cost approximation, the Q-function becomes quadratic:

\[
Q(\delta x_t,\delta u_t)
\approx
Q_0
+
Q_x^T\delta x_t
+
Q_u^T\delta u_t
+
\frac{1}{2}\delta x_t^T Q_{xx}\delta x_t
+
\delta u_t^T Q_{ux}\delta x_t
+
\frac{1}{2}\delta u_t^T Q_{uu}\delta u_t
\]

The Q-function derivatives are:

\[
Q_x = \ell_x + f_x^T V_x
\]

\[
Q_u = \ell_u + f_u^T V_x
\]

\[
Q_{xx} = \ell_{xx} + f_x^T V_{xx}f_x
\]

\[
Q_{ux} = \ell_{ux} + f_u^T V_{xx}f_x
\]

\[
Q_{uu} = \ell_{uu} + f_u^T V_{xx}f_u
\]

where \(V_x\) and \(V_{xx}\) are the gradient and Hessian of the value function at the next time step.

---

## 6. Derivative With Respect to the Control Deviation

To find the optimal local control update, minimize the local Q-function with respect to \(\delta u_t\).

The terms involving \(\delta u_t\) are:

\[
Q(\delta x_t,\delta u_t)
=
Q_u^T\delta u_t
+
\delta u_t^T Q_{ux}\delta x_t
+
\frac{1}{2}\delta u_t^T Q_{uu}\delta u_t
+
\text{terms not depending on } \delta u_t
\]

Take the derivative with respect to \(\delta u_t\):

\[
\frac{\partial Q}{\partial \delta u_t}
=
Q_u
+
Q_{ux}\delta x_t
+
Q_{uu}\delta u_t
\]

Set the derivative equal to zero:

\[
Q_u
+
Q_{ux}\delta x_t
+
Q_{uu}\delta u_t
=
0
\]

Solve for \(\delta u_t\):

\[
Q_{uu}\delta u_t
=
-
Q_u
-
Q_{ux}\delta x_t
\]

\[
\delta u_t^*
=
-
Q_{uu}^{-1}Q_u
-
Q_{uu}^{-1}Q_{ux}\delta x_t
\]

---

## 7. iLQR Feedforward and Feedback Gains

Define:

\[
k_t =
-
Q_{uu}^{-1}Q_u
\]

\[
K_t =
-
Q_{uu}^{-1}Q_{ux}
\]

Then the optimal local control update is:

\[
\delta u_t^*
=
k_t
+
K_t\delta x_t
\]

Therefore, the updated control law is:

\[
u_t^{new}
=
\bar{u}_t
+
\alpha k_t
+
K_t(x_t^{new}-\bar{x}_t)
\]

where:

- \(k_t\) is the feedforward update
- \(K_t\) is the feedback gain
- \(\alpha\) is the line-search step size
- \(\bar{x}_t,\bar{u}_t\) are the nominal state and control

---

## 8. Value Function Update

After finding the optimal local control update, the value function derivatives are updated backward.

The value gradient is:

\[
V_x
=
Q_x
+
K_t^T Q_u
+
Q_{xu}k_t
+
K_t^TQ_{uu}k_t
\]

The value Hessian is:

\[
V_{xx}
=
Q_{xx}
+
K_t^TQ_{ux}
+
Q_{xu}K_t
+
K_t^TQ_{uu}K_t
\]

This backward recursion is repeated from \(T\) to \(0\).

---

## 9. Terminal Condition

At the terminal time, there is no future cost left, so the value function equals the terminal cost.

\[
V_T(x_T)=\ell_f(x_T)
\]

If the terminal cost is quadratic:

\[
\ell_f(x_T)
=
\frac{1}{2}(x_T-x_d)^TQ_f(x_T-x_d)
\]

then:

\[
V_x(T)=Q_f(x_T-x_d)
\]

\[
V_{xx}(T)=Q_f
\]

> **Explanation: Terminal condition**
>
> At the final time \(T\), there is no future cost, so the value derivatives come directly from the terminal cost.
>
> \[
> V_x(T)=\ell_{f,x}
> \]
>
> \[
> V_{xx}(T)=\ell_{f,xx}
> \]

---

## 10. iLQR Algorithm Summary

iLQR repeats the following steps:

1. Start with an initial control trajectory:

\[
\bar{u}_0,\bar{u}_1,\dots,\bar{u}_{T-1}
\]

2. Forward rollout the nonlinear dynamics:

\[
\bar{x}_{t+1}=f(\bar{x}_t,\bar{u}_t)
\]

3. Linearize the dynamics around the nominal trajectory:

\[
\delta x_{t+1}
\approx
f_x\delta x_t+f_u\delta u_t
\]

4. Build the local quadratic Q-function.

5. Backward pass to compute:

\[
k_t=-Q_{uu}^{-1}Q_u
\]

\[
K_t=-Q_{uu}^{-1}Q_{ux}
\]

6. Forward pass with updated control:

\[
u_t^{new}
=
\bar{u}_t
+
\alpha k_t
+
K_t(x_t^{new}-\bar{x}_t)
\]

7. Repeat until the total cost stops decreasing.

---

## Summary

Regular LQR solves a linear-quadratic problem directly:

\[
x_{t+1}=Ax_t+Bu_t
\]

iLQR solves a nonlinear problem by repeatedly creating local LQR approximations:

\[
x_{t+1}=f(x_t,u_t)
\]

At each iteration, iLQR linearizes the dynamics:

\[
\delta x_{t+1}
\approx
f_x\delta x_t+f_u\delta u_t
\]

and approximates the cost quadratically.

The key derivative is:

\[
\frac{\partial Q}{\partial \delta u_t}
=
Q_u
+
Q_{ux}\delta x_t
+
Q_{uu}\delta u_t
\]

Setting it equal to zero gives:

\[
\delta u_t^*
=
-
Q_{uu}^{-1}Q_u
-
Q_{uu}^{-1}Q_{ux}\delta x_t
\]

or:

\[
\delta u_t^*
=
k_t+K_t\delta x_t
\]

where:

\[
k_t=-Q_{uu}^{-1}Q_u
\]

\[
K_t=-Q_{uu}^{-1}Q_{ux}
\]