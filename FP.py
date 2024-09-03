def FP_grid(ct_spec):
    d_beta = np.pi * 360 / ct_spec["view"] / 180  # angular step size in radian
    beta = -1 * torch.linspace(
        0,
        (ct_spec["view"] - 1) * d_beta,
        ct_spec["view"],
    )
    ct_spec["DCD"] = ct_spec["SDD"] - ct_spec["SCD"]

    range_det = torch.tensor(
        [
            -ct_spec["det_interval"] * (ct_spec["num_det"] - 1) / 2,
            ct_spec["det_interval"] * (ct_spec["num_det"] - 1) / 2,
        ],
    )

    axis_x = torch.linspace(-ct_spec["recon_size"][0] / 2, ct_spec["recon_size"][0] / 2, ct_spec["recon_size"][0] + 1) * ct_spec["recon_interval"]
    axis_y = torch.linspace(-ct_spec["recon_size"][1] / 2, ct_spec["recon_size"][1] / 2, ct_spec["recon_size"][1] + 1) * ct_spec["recon_interval"]

    center = (ct_spec["num_det"] + 1) / 2
    delta_gamma = ct_spec["det_interval"] / ct_spec["SDD"]
    gamma_vec = (torch.linspace(1, ct_spec["num_det"], ct_spec["num_det"]) - center) * delta_gamma
    range_det_x = ct_spec["SDD"] * torch.sin(gamma_vec)  # (n_det)
    range_det_y = -(ct_spec["SDD"] * torch.cos(gamma_vec) - ct_spec["SCD"])

    src_zero = torch.tensor([0, ct_spec["SCD"]])
    src_point = torch.cat(
        (
            src_zero[0] * torch.cos(beta)[None, :] - src_zero[1] * torch.sin(beta)[None, :],
            +src_zero[0] * torch.sin(beta)[None, :] + src_zero[1] * torch.cos(beta)[None, :],
        ),
        dim=0,
    )  # (2, view)

    det_x_rot = torch.cos(beta)[:, None] * range_det_x[None, :] - torch.sin(beta)[:, None] * range_det_y[None, :]  # (view, n_det)
    det_y_rot = torch.sin(beta)[:, None] * range_det_x[None, :] + torch.cos(beta)[:, None] * range_det_y[None, :]  # (view, n_det)

    ax = (axis_x[None, None, :] - src_point[0][:, None, None]) / (det_x_rot - src_point[0, :][:, None])[:, :, None]  # (view, det, grid+1)
    ay = (axis_y[None, None, :] - src_point[1][:, None, None]) / (det_y_rot - src_point[1, :][:, None])[:, :, None]

    a_min = torch.maximum(
        torch.minimum(ax[:, :, 0], ax[:, :, -1]),
        torch.minimum(ay[:, :, 0], ay[:, :, -1]),
    )  # (view, det)
    a_min[a_min < 0] = 0
    a_max = torch.minimum(
        torch.maximum(ax[:, :, 0], ax[:, :, -1]),
        torch.maximum(ay[:, :, 0], ay[:, :, -1]),
    )  # (view, det)
    a_max[a_max > 1] = 1

    axy = torch.cat([ax, ay], dim=2)
    axy[torch.logical_or(axy > a_max[:, :, None], axy < a_min[:, :, None])] = float("nan")
    axy, _ = torch.sort(axy, dim=2)
    diff_axy = torch.diff(axy, n=1, dim=2)  # shape(view, det, x+y+1)
    diff_axy[torch.isnan(diff_axy)] = 0
    a_mid = axy[:, :, :-1] + diff_axy / 2  # shape(view, det, x+y+1)
    a_mid[torch.isnan(a_mid)] = 0
    s2d = torch.sqrt(torch.pow(det_x_rot - src_point[0][:, None], 2) + torch.pow(det_y_rot - src_point[1][:, None], 2))  # (view, det)
    weighting = diff_axy[None, None, :, :, :] * s2d[None, None, :, :, None]  # (batch, ch, view, det, x+y+1)
    x_pos = a_mid * (det_x_rot[:, :, None] - src_point[0, :][:, None, None]) + src_point[0, :][:, None, None]  # (view, det, x+y+1)
    x_pos = x_pos / (ct_spec["recon_interval"] * (ct_spec["recon_size"][0]) / 2)  # normalize grid
    y_pos = a_mid * (det_y_rot[:, :, None] - src_point[1, :][:, None, None]) + src_point[1, :][:, None, None]
    y_pos = y_pos / (ct_spec["recon_interval"] * (ct_spec["recon_size"][1]) / 2)

    grid_pos = torch.cat((x_pos.unsqueeze(-1), y_pos.unsqueeze(-1)), dim=3)  # (view, det, x+y+1, 2)
    return grid_pos, weighting
