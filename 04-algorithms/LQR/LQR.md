# LQR Derivative and Cost-to-Go Function

For a discrete-time linear system:

\[
x_{t+1} = Ax_t + Bu_t
\]

where:

- \(x_t\) is the state at time \(t\)
- \(u_t\) is the control input at time \(t\)
- \(A\) describes the system dynamics
- \(B\) describes how the control input affects the system

The goal of LQR is to find an optimal control input \(u_t\) that minimizes a quadratic cost.

---

## 1. LQR Objective Function

The total cost from time \(t\) to the final time \(T\) is:

\[
J_t = \sum_{k=t}^{T-1} \left(x_k^T Q x_k + u_k^T R u_k\right) + x_T^T Q_f x_T
\]

where:

- \(x_k^T Q x_k\) penalizes large state errors
- \(u_k^T R u_k\) penalizes large control effort
- \(x_T^T Q_f x_T\) is the final terminal cost
- \(Q \succeq 0\), meaning \(Q\) is positive semi-definite
- \(R \succ 0\), meaning \(R\) is positive definite

The objective is:

\[
\min_{u_t, u_{t+1}, \dots, u_{T-1}} J_t
\]

---

## 2. How the Cost-to-Go Function Comes From Dynamic Programming

Instead of minimizing the whole future cost at once, LQR uses dynamic programming.

Define the cost-to-go function:

\[
V_t(x_t)
\]

This means: the minimum possible future cost starting from state \(x_t\) at time \(t\).

So:

\[
V_t(x_t)
=
\min_{u_t, u_{t+1}, \dots, u_{T-1}}
\left[
\sum_{k=t}^{T-1}
\left(x_k^T Q x_k + u_k^T R u_k\right)
+
x_T^T Q_f x_T
\right]
\]

Using dynamic programming, we separate the current-step cost from the future cost:

\[
V_t(x_t)
=
\min_{u_t}
\left[
x_t^T Q x_t
+
u_t^T R u_t
+
V_{t+1}(x_{t+1})
\right]
\]

Since the system dynamics are:

\[
x_{t+1}=Ax_t+Bu_t
\]

we can write:

\[
V_t(x_t)
=
\min_{u_t}
\left[
x_t^T Q x_t
+
u_t^T R u_t
+
V_{t+1}(Ax_t+Bu_t)
\right]
\]

---

## 3. Quadratic Form of the Cost-to-Go Function

Because the system is linear and the cost is quadratic, the optimal cost-to-go is also quadratic.

Assume:

\[
V_{t+1}(x_{t+1})
=
x_{t+1}^T P_{t+1} x_{t+1}
\]

where \(P_{t+1}\) is a symmetric positive semi-definite matrix.

Substitute this into the dynamic programming equation:

\[
V_t(x_t)
=
\min_{u_t}
\left[
x_t^T Q x_t
+
u_t^T R u_t
+
x_{t+1}^T P_{t+1} x_{t+1}
\right]
\]

Using:

\[
x_{t+1}=Ax_t+Bu_t
\]

we get:

\[
V_t(x_t)
=
\min_{u_t}
\left[
x_t^T Q x_t
+
u_t^T R u_t
+
(Ax_t+Bu_t)^T P_{t+1}(Ax_t+Bu_t)
\right]
\]

So the one-step cost-to-go expression is:

\[
J_t
=
x_t^T Q x_t
+
u_t^T R u_t
+
(Ax_t+Bu_t)^T P_{t+1}(Ax_t+Bu_t)
\]

---

## 4. Derivative With Respect to the Control Input

Now we minimize \(J_t\) with respect to \(u_t\).

\[
J_t
=
x_t^T Q x_t
+
u_t^T R u_t
+
(Ax_t+Bu_t)^T P_{t+1}(Ax_t+Bu_t)
\]

The first term does not depend on \(u_t\), so:

\[
\frac{\partial}{\partial u_t}
\left(x_t^T Q x_t\right)
=
0
\]

For the control cost term:

\[
\frac{\partial}{\partial u_t}
\left(u_t^T R u_t\right)
=
2Ru_t
\]

For the future cost term:

\[
\frac{\partial}{\partial u_t}
\left[
(Ax_t+Bu_t)^T P_{t+1}(Ax_t+Bu_t)
\right]
=
2B^T P_{t+1}(Ax_t+Bu_t)
\]

Therefore:

\[
\frac{\partial J_t}{\partial u_t}
=
2Ru_t
+
2B^T P_{t+1}(Ax_t+Bu_t)
\]

---

## 5. Set the Derivative Equal to Zero

To find the minimizing control input, set the derivative equal to zero:

\[
2Ru_t
+
2B^T P_{t+1}(Ax_t+Bu_t)
=
0
\]

Divide both sides by 2:

\[
Ru_t
+
B^T P_{t+1}(Ax_t+Bu_t)
=
0
\]

Expand the second term:

\[
Ru_t
+
B^T P_{t+1}Ax_t
+
B^T P_{t+1}Bu_t
=
0
\]

Group the terms involving \(u_t\):

\[
(R+B^T P_{t+1}B)u_t
+
B^T P_{t+1}Ax_t
=
0
\]

Move the state term to the other side:

\[
(R+B^T P_{t+1}B)u_t
=
-
B^T P_{t+1}Ax_t
\]

Multiply both sides by \((R+B^T P_{t+1}B)^{-1}\):

\[
u_t^*
=
-
(R+B^T P_{t+1}B)^{-1}
B^T P_{t+1}A x_t
\]

---

## 6. LQR Feedback Gain

Define the feedback gain matrix:

\[
K_t
=
(R+B^T P_{t+1}B)^{-1}
B^T P_{t+1}A
\]

Then the optimal control law becomes:

\[
u_t^* = -K_t x_t
\]

So the LQR controller is:

\[
\boxed{
u_t^*
=
-
(R+B^T P_{t+1}B)^{-1}
B^T P_{t+1}A x_t
}
\]

---

## 7. Riccati Recursion

After finding the optimal \(u_t^*\), the matrix \(P_t\) can be updated backward in time:

\[
P_t
=
Q
+
A^T P_{t+1}A
-
A^T P_{t+1}B
(R+B^T P_{t+1}B)^{-1}
B^T P_{t+1}A
\]

This is called the discrete-time Riccati recursion.

The terminal condition is:

\[
P_T = Q_f
\]

> **Explanation: Why \(P_T = Q_f\)**
>
> At the final time \(T\), there is no future cost left, so the cost-to-go equals only the terminal cost.
>
> \[ V_T(x_T) = x_T^T P_T x_T \]
>
> \[ V_T(x_T) = x_T^T Q_f x_T \]
>
> Therefore:
>
> \[ x_T^T P_T x_T = x_T^T Q_f x_T \]
>
> \[ \boxed{P_T = Q_f} \]

So LQR solves backward from \(T\) to \(0\), then applies the control law forward in time:

\[
u_t^* = -K_t x_t
\]

---

## Summary

The cost-to-go function comes from dynamic programming:

\[
V_t(x_t)
=
\min_{u_t}
\left[
x_t^TQx_t
+
u_t^TRu_t
+
V_{t+1}(x_{t+1})
\right]
\]

Because the dynamics are linear and the cost is quadratic, we assume:

\[
V_{t+1}(x_{t+1})
=
x_{t+1}^TP_{t+1}x_{t+1}
\]

Substituting the dynamics gives:

\[
J_t
=
x_t^TQx_t
+
u_t^TRu_t
+
(Ax_t+Bu_t)^TP_{t+1}(Ax_t+Bu_t)
\]

Taking the derivative with respect to \(u_t\), setting it equal to zero, and solving gives:

\[
u_t^*
=
-
(R+B^TP_{t+1}B)^{-1}
B^TP_{t+1}Ax_t
\]

or simply:

\[
u_t^* = -K_t x_t
\]