import torch
import torch.nn.functional as F


def copy_face_open_bc(uz, uy, ux):
    # uz: (D+1,H,W), uy: (D,H+1,W), ux: (D,H,W+1)
    uz = uz.clone()
    uy = uy.clone()
    ux = ux.clone()

    uz = torch.cat([uz[1:2], uz[1:-1], uz[-2:-1]], dim=0)
    uy = torch.cat([uy[:,1:2,:], uy[:,1:-1,:], uy[:,-2:-1,:]], dim=1)
    ux = torch.cat([ux[:,:,1:2], ux[:,:,1:-1], ux[:,:,-2:-1]], dim=2)
    return uz, uy, ux



def velocity_centers_to_faces_open(u):
    D, H, W, _ = u.shape
    uz = torch.zeros((D + 1, H, W), device=u.device, dtype=u.dtype)
    uy = torch.zeros((D, H + 1, W), device=u.device, dtype=u.dtype)
    ux = torch.zeros((D, H, W + 1), device=u.device, dtype=u.dtype)

    uz = uz.clone(); uy = uy.clone(); ux = ux.clone()
    uz[1:D,:,:] = 0.5*(u[0:D-1,:,:,0] + u[1:D,:,:,0])
    uy[:,1:H,:] = 0.5*(u[:,0:H-1,:,1] + u[:,1:H,:,1])
    ux[:,:,1:W] = 0.5*(u[:,:,0:W-1,2] + u[:,:,1:W,2])

    uz, uy, ux = copy_face_open_bc(uz, uy, ux)
    return uz, uy, ux


def faces_to_velocity_centers(uz, uy, ux):
    """
    Convert face-normal velocities back to cell-centered u by averaging adjacent faces.
    """
    Dp1, H, W = uz.shape
    D = Dp1 - 1
    _, Hp1, _ = uy.shape
    H = Hp1 - 1
    _, _, Wp1 = ux.shape
    W = Wp1 - 1

    device = uz.device
    dtype = uz.dtype
    u = torch.zeros((D, H, W, 3), device=device, dtype=dtype)

    # center velocity is average of its two adjacent faces along each axis
    u[:, :, :, 0] = 0.5 * (uz[0:D, :, :] + uz[1:D+1, :, :])
    u[:, :, :, 1] = 0.5 * (uy[:, 0:H, :] + uy[:, 1:H+1, :])
    u[:, :, :, 2] = 0.5 * (ux[:, :, 0:W] + ux[:, :, 1:W+1])

    return u


def divergence_from_faces(uz, uy, ux, dx=1.0):
    """
    Divergence at cell centers from face-normal velocities.
    div[z,y,x] = (uz[z+1]-uz[z] + uy[y+1]-uy[y] + ux[x+1]-ux[x]) / dx
    """
    invdx = 1.0 / dx
    div = (
        (uz[1:, :, :] - uz[:-1, :, :]) +
        (uy[:, 1:, :] - uy[:, :-1, :]) +
        (ux[:, :, 1:] - ux[:, :, :-1])
    ) * invdx
    return div


def grad_p_to_faces(p, dx=1.0):
    """
    Pressure gradient on faces (forward differences from cell centers).
    Returns gz (D+1,H,W), gy (D,H+1,W), gx (D,H,W+1).
    Boundary faces are 0.
    """
    D, H, W = p.shape
    device = p.device
    dtype = p.dtype
    invdx = 1.0 / dx

    gz = torch.zeros((D + 1, H, W), device=device, dtype=dtype)
    gy = torch.zeros((D, H + 1, W), device=device, dtype=dtype)
    gx = torch.zeros((D, H, W + 1), device=device, dtype=dtype)

    gz[1:D, :, :] = (p[1:D, :, :] - p[0:D-1, :, :]) * invdx
    gy[:, 1:H, :] = (p[:, 1:H, :] - p[:, 0:H-1, :]) * invdx
    gx[:, :, 1:W] = (p[:, :, 1:W] - p[:, :, 0:W-1]) * invdx

    return gz, gy, gx


### STANDARD PROJECTION (no variable coefficients) ###

def laplace_neumann(p, dx=1.0):
    """
    Neumann (zero-normal-derivative) Laplacian using replicate padding.
    p: (D,H,W)
    returns: (D,H,W)
    """
    # Add batch+channel dims: (1,1,D,H,W)
    pp = p.unsqueeze(0).unsqueeze(0)
    # Pad order for 3D: (W_left, W_right, H_left, H_right, D_left, D_right)
    pp = F.pad(pp, (1,1, 1,1, 1,1), mode='replicate')
    pp = pp[0,0]  # back to (D+2,H+2,W+2)

    c = pp[1:-1, 1:-1, 1:-1]
    lap = (
        pp[2:  , 1:-1, 1:-1] + pp[:-2 , 1:-1, 1:-1] +
        pp[1:-1, 2:  , 1:-1] + pp[1:-1, :-2 , 1:-1] +
        pp[1:-1, 1:-1, 2:  ] + pp[1:-1, 1:-1, :-2 ] -
        6.0 * c
    ) / (dx * dx)
    return lap

def pcg_solve_laplace(rhs, p0=None, iters=50, dx=1.0, tol=1e-6):
    p = torch.zeros_like(rhs) if p0 is None else p0.clone()

    # diagonal preconditioner for -Laplace with this stencil
    diag = torch.full_like(rhs, 6.0 / (dx * dx))

    # pin one cell to kill constant nullspace (gauge)
    pin_mask = torch.zeros_like(rhs)
    pin_mask = pin_mask.clone()  # keep autograd happy
    pin_mask = pin_mask + torch.zeros_like(rhs)  # no-op, but ensures new tensor
    pin_mask = pin_mask.index_put((torch.tensor([0], device=rhs.device),
                                   torch.tensor([0], device=rhs.device),
                                   torch.tensor([0], device=rhs.device)),
                                  torch.tensor([1.0], device=rhs.device),
                                  accumulate=False)

    def apply_A(x):
        y = laplace_neumann(x, dx=dx)
        # enforce y[pinned] = x[pinned] out-of-place
        return y + pin_mask * (x - y)

    # initial residual
    r = rhs - apply_A(p)
    r = r * (1.0 - pin_mask)

    z = r / diag
    d = z
    rz_old = torch.sum(r * z)

    # If rhs is zero (or near), exit early
    if torch.abs(rz_old) < 1e-30:
        return p

    for _ in range(iters):
        Ad = apply_A(d)
        denom = torch.sum(d * Ad) + 1e-30
        alpha = rz_old / denom

        p = p + alpha * d
        r = r - alpha * Ad
        r = r * (1.0 - pin_mask)

        if torch.norm(r) < tol:
            break

        z = r / diag
        rz_new = torch.sum(r * z)

        beta = rz_new / (rz_old + 1e-30)
        d = z + beta * d
        rz_old = rz_new

    return p


def cg_solve(A_mv, b, x0=None, M_inv=None, iters=50, tol=0.0):
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A_mv(x)
    z = r if M_inv is None else M_inv(r)
    p = z
    rz = torch.sum(r * z)

    if torch.abs(rz) < 1e-30:
        return x

    for _ in range(iters):
        Ap = A_mv(p)
        alpha = rz / (torch.sum(p * Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        if tol > 0 and torch.linalg.norm(r) < tol:
            break
        z = r if M_inv is None else M_inv(r)
        rz_new = torch.sum(r * z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new
    return x

def make_pin_mask(shape, device, dtype):
    pin = torch.zeros(shape, device=device, dtype=dtype)
    idx = (torch.tensor([0], device=device),
           torch.tensor([0], device=device),
           torch.tensor([0], device=device))
    pin = pin.index_put(idx, torch.tensor([1.0], device=device, dtype=dtype), accumulate=False)
    return pin

def apply_A_neumann(x, dx, pin_mask):
    y = laplace_neumann(x, dx=dx)
    # enforce y[pinned] = x[pinned] without slicing
    y = y + pin_mask * (x - y)
    return y


class PressureSolveNeumann(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, dx: float, iters: int):
        # Save for backward
        pin_mask = make_pin_mask(rhs.shape, rhs.device, rhs.dtype)
        ctx.save_for_backward(pin_mask)
        ctx.dx = float(dx)
        ctx.iters = int(iters)

        # Preconditioner (Jacobi)
        diag = torch.full_like(rhs, 6.0 / (dx * dx))
        M_inv = lambda r: r / (diag + 1e-8)

        with torch.no_grad():
            A_mv = lambda x: apply_A_neumann(x, ctx.dx, pin_mask)
            # IMPORTANT: enforce pin constraint on rhs too (so system is consistent)
            rhs_eff = rhs * (1.0 - pin_mask)
            p = cg_solve(A_mv, rhs_eff, x0=None, M_inv=M_inv, iters=ctx.iters, tol=0.0)

        return p

    @staticmethod
    def backward(ctx, grad_p):
        (pin_mask,) = ctx.saved_tensors
        dx = ctx.dx
        iters = ctx.iters

        diag = torch.full_like(grad_p, 6.0 / (dx * dx))
        M_inv = lambda r: r / (diag + 1e-8)

        with torch.no_grad():
            A_mv = lambda x: apply_A_neumann(x, dx, pin_mask)
            grad_eff = grad_p * (1.0 - pin_mask)
            lam = cg_solve(A_mv, grad_eff, x0=None, M_inv=M_inv, iters=iters, tol=0.0)

        # Gradient wrt rhs is λ. No grads for dx/iters.
        return lam, None, None

def pressure_solve_neumann(rhs, dx=1.0, iters=50):
    return PressureSolveNeumann.apply(rhs, float(dx), int(iters))



def project_standard(u, iters=50, dx=1.0):
    uz, uy, ux = velocity_centers_to_faces_open(u)
    div_u = divergence_from_faces(uz, uy, ux, dx=dx)

    p = pressure_solve_neumann(div_u, dx=dx, iters=iters)

    gz, gy, gx = grad_p_to_faces(p, dx=dx)
    uz = uz - gz
    uy = uy - gy
    ux = ux - gx

    uz, uy, ux = copy_face_open_bc(uz, uy, ux)
    return faces_to_velocity_centers(uz, uy, ux), p
