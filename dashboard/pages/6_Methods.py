import streamlit as st

st.title("How each method works")

st.markdown(
    """
## Kalman filter (KF)

```
pixels from ≥2 cameras
        ↓
multi-ray triangulation
        ↓
3D position measurement z_k
        ↓
linear Kalman update with H = [I₃ | 0]
```

**Strength:** simple, fast, stable when triangulation geometry is good.

**Limitation:** needs at least two cameras; ignores raw pixel geometry after triangulation.

---

## Extended Kalman filter (EKF)

```
predict 3D state x_k
        ↓
project to each camera: h(x_k)
        ↓
Jacobian H_k = ∂h/∂x
        ↓
sequential pixel updates + NIS gating
```

**Strength:** uses every camera independently; works with one camera for short intervals.

**Limitation:** linearisation error; projection can become invalid behind the camera.

---

## Unscented Kalman filter (UKF)

```
predict sigma points from (x_k, P_k)
        ↓
propagate each sigma point through h(x)
        ↓
reconstruct predicted measurement mean/covariance
        ↓
UKF correction + innovation gating
```

**Strength:** better handles nonlinear projection than EKF.

**Limitation:** more expensive; invalid sigma points behind cameras must be rejected.
"""
)
