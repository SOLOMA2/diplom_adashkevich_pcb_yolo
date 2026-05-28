from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)


def size(
    xywh: torch.Tensor,
    classes: torch.Tensor,
    img_inspected: torch.Tensor,
    size_class: torch.Tensor,
    defect_names: Sequence[str],
    *,
    tol: int = 5,
    last_seen_window: int = 8,
    skip_alert_class_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    """Track inferred defect extent; sustained growth fires a warning."""
    if skip_alert_class_ids is None:
        skip_alert_class_ids = frozenset({0})
    device, dtype = xywh.device, xywh.dtype
    nc = len(defect_names)

    xywh = xywh.reshape(-1, 4).to(device=device, dtype=dtype)
    classes = classes.reshape(-1).to(device=device, dtype=torch.long)
    img_inspected = img_inspected.reshape(-1).to(device=device, dtype=dtype)
    row = xywh.shape[0]

    pred_1: torch.Tensor
    if row == 0:
        pred_1 = torch.empty(0, 2, device=device, dtype=dtype)
    else:
        parts = []
        for i in range(row):
            c = int(classes[i].item())
            w, h = xywh[i, 2], xywh[i, 3]
            if w <= 0 or h <= 0:
                sz_tmp = torch.tensor(0.0, device=device, dtype=dtype)
            elif c == 0:
                sz_tmp = (w + h) / 2
            else:
                rel = w / h
                if rel < 0.5:
                    sz_tmp = h
                elif rel > 1.5:
                    sz_tmp = w
                else:
                    sz_tmp = torch.sqrt(w * w + h * h)
            parts.append(torch.stack([classes[i].to(dtype=dtype), sz_tmp]))
        pred_1 = torch.stack(parts, dim=0)

    z = torch.zeros((), device=device, dtype=dtype)
    param_old = size_class[:, 1].clone()
    param_new = []
    for j in range(nc):
        det_tmp = pred_1[:, 0] == j if pred_1.numel() else torch.zeros(0, dtype=torch.bool, device=device)
        if pred_1.numel() and torch.any(det_tmp):
            max_tmp = pred_1[:, 1][det_tmp].max().unsqueeze(0)
        else:
            max_tmp = z.unsqueeze(0)
        param_new.append(max_tmp.squeeze(0))
    param_new = torch.stack(param_new)
    up = (param_new > param_old).to(dtype)

    size_ls_list = []
    for m in range(nc):
        if pred_1.numel():
            nm = pred_1[:, 0] == m
            if torch.any(nm):
                ls = img_inspected.reshape(-1)[nm].max().unsqueeze(0)
            else:
                ls = size_class[m, 3].unsqueeze(0)
        else:
            ls = size_class[m, 3].unsqueeze(0)
        size_ls_list.append(ls.squeeze(0))
    size_ls = torch.stack(size_ls_list)

    for n in range(nc):
        if size_ls[n] - size_class[n, 3] < last_seen_window:
            size_class[n, 2] = size_class[n, 2] + up[n]
            size_class[n, 1] = param_new[n] if up[n] == 1 else param_old[n]
            if size_class[n, 2] >= tol and up[n] != 0 and n not in skip_alert_class_ids:
                logger.warning("Defect: %s is increasingly bigger.", defect_names[n])
        else:
            size_class[n, 2] = up[n]
            size_class[n, 1] = 0

    size_class[:, 3] = size_ls
    return size_class


def rep(
    img_frame_ids: torch.Tensor | None,
    xywh: torch.Tensor,
    occ_point: torch.Tensor,
    *,
    max_points: int = 500,
    tol: float = 1.0,
    rep_thr: int = 3,
    last: float = 5.0,
) -> torch.Tensor:
    """
    Spatial recurrence near fixed centroids. ``img_frame_ids`` must have one
    entry per detection (frame index).
    """
    if img_frame_ids is None or xywh.numel() == 0:
        return occ_point

    device = xywh.device
    dtype = xywh.dtype
    occ_point = occ_point.to(device=device, dtype=dtype)
    xywh_flat = xywh.reshape(-1, 4)
    img_frame_ids = img_frame_ids.reshape(-1).to(device=device, dtype=dtype)
    row = xywh_flat.shape[0]
    if row == 0:
        return occ_point

    if occ_point.numel() == 0:
        xy = xywh_flat[:, 0:2]
        fr = img_frame_ids.reshape(row, 1)
        one = torch.ones(row, 1, device=device, dtype=dtype)
        return torch.cat([xy, fr, one], dim=1)

    if occ_point.shape[0] > max_points:
        occ_point = occ_point[-max_points:, :]

    row_occ = occ_point.shape[0]
    occ_point = occ_point.reshape(row_occ, 4)
    xy_mid = xywh_flat[:, 0:2]
    frame_flat = img_frame_ids.reshape(row)

    pot_rows = []
    npot_rows = []
    for i_row in occ_point[:, 0:2]:
        for j_idx in range(len(xy_mid)):
            dist = torch.hypot(i_row[0] - xy_mid[j_idx, 0], i_row[1] - xy_mid[j_idx, 1])
            if bool(dist <= tol):
                fid = frame_flat[j_idx]
                pot_rows.append(torch.tensor([i_row[0].item(), i_row[1].item(), fid.item()], device=device, dtype=dtype))
                pot_rows.append(torch.tensor([xy_mid[j_idx, 0].item(), xy_mid[j_idx, 1].item(), fid.item()], device=device, dtype=dtype))
            else:
                npot_rows.append(torch.tensor([xy_mid[j_idx, 0].item(), xy_mid[j_idx, 1].item(), -1.0], device=device, dtype=dtype))

    pot_list = torch.stack(pot_rows, dim=0) if pot_rows else torch.empty(0, 3, device=device, dtype=dtype)
    npot_list = torch.stack(npot_rows, dim=0) if npot_rows else torch.empty(0, 3, device=device, dtype=dtype)

    if pot_list.shape[0] == 0 and npot_list.shape[0] == 0:
        return occ_point

    ind = torch.empty(0, 2, device=device, dtype=dtype)
    ind_cnts = torch.empty(0, device=device, dtype=dtype)
    ind_ls = torch.empty(0, device=device, dtype=dtype)

    if pot_list.shape[0] > 0:
        ind, ind_cnts = torch.unique(pot_list[:, 0:2], dim=0, return_counts=True)
        ind_ls_list = []
        for k_idx in range(ind.shape[0]):
            k = ind[k_idx]
            mask = torch.all(torch.isclose(pot_list[:, 0:2], k.unsqueeze(0), rtol=0.0, atol=0.0), dim=1)
            ls_min = pot_list[mask][:, 2].min(dim=0).values.unsqueeze(0)
            ind_ls_list.append(ls_min)
        ind_ls = torch.cat(ind_ls_list, dim=0)

    if npot_list.shape[0] > 0:
        ind = torch.cat([ind, npot_list[:, 0:2]], dim=0)
        pad = torch.zeros(npot_list.shape[0], device=device, dtype=dtype)
        ind_cnts = torch.cat([ind_cnts, pad], dim=0)
        ind_ls = torch.cat([ind_ls, npot_list[:, 2]], dim=0)

    for m in range(ind.shape[0]):
        pos = torch.all(torch.isclose(occ_point[:, 0:2], ind[m].unsqueeze(0), rtol=0.0, atol=0.0), dim=1).nonzero(as_tuple=True)[
            0
        ]
        if pos.numel() != 0:
            p = pos[0]
            if ind_ls[m] - occ_point[p, 2] >= last:
                occ_point[p, 2] = ind_ls[m]
                occ_point[p, 3] = ind_cnts[m]
            else:
                occ_point[p, 3] = occ_point[p, 3] + ind_cnts[m]
                if occ_point[p, 3] >= rep_thr:
                    logger.warning("Cluster appearing at x = %.3f, y = %.3f.", float(occ_point[p, 0]), float(occ_point[p, 1]))
        else:
            row_new = torch.cat([ind[m], ind_ls[m].unsqueeze(0), ind_cnts[m].unsqueeze(0)], dim=0)
            occ_point = torch.cat([occ_point, row_new.unsqueeze(0)], dim=0)

    return occ_point


def cnt(
    classes: torch.Tensor,
    img_inspected: torch.Tensor,
    cnt_class: torch.Tensor,
    defect_names: Sequence[str],
    *,
    tol: int = 5,
    last_seen_window: int = 6,
    skip_alert_class_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    """
    Per-class defect rate trend. Uses the same ``unique + offset`` counting
    trick as the reference implementation.
    """
    if skip_alert_class_ids is None:
        skip_alert_class_ids = frozenset({0})
    device = classes.device
    dtype = cnt_class.dtype
    nc = len(defect_names)

    classes = classes.reshape(-1).to(device=device, dtype=torch.long)
    img_inspected = img_inspected.reshape(-1).to(device=device, dtype=dtype)

    no_img_inspected = (
        img_inspected.max() - img_inspected.min() + 1 if img_inspected.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
    )

    class_cnt = torch.arange(nc, device=device, dtype=torch.long)
    classes_tmp = torch.cat([class_cnt, classes], dim=0)
    ind, ind_cnts = torch.unique(classes_tmp, return_counts=True)
    ind_cnts = ind_cnts.to(dtype=dtype) - 1

    full_counts = torch.zeros(nc, device=device, dtype=dtype)
    for k in range(ind.shape[0]):
        cid = int(ind[k].item())
        if 0 <= cid < nc:
            full_counts[cid] = ind_cnts[k]

    denom_old = cnt_class[:, 2].clamp_min(1.0)
    param_old = cnt_class[:, 1] / denom_old
    param_new = (cnt_class[:, 1] + full_counts) / (cnt_class[:, 2] + no_img_inspected)
    up = (param_new > param_old).to(dtype)

    cnt_ls = torch.empty(nc, device=device, dtype=dtype)
    for j in range(nc):
        nm = classes == j
        if classes.numel() and torch.any(nm):
            cnt_ls[j] = img_inspected[nm].max()
        else:
            cnt_ls[j] = cnt_class[j, 4]

        if cnt_ls[j] - cnt_class[j, 4] < last_seen_window:
            cnt_class[j, 3] = cnt_class[j, 3] + up[j]
            if cnt_class[j, 3] >= tol and j not in skip_alert_class_ids and up[j] != 0:
                logger.warning("Defect: %s is increasing.", defect_names[j])
        else:
            cnt_class[j, 3] = up[j]

    cnt_class[:, 1] = cnt_class[:, 1] + full_counts
    cnt_class[:, 2] = cnt_class[:, 2] + no_img_inspected
    cnt_class[:, 4] = cnt_ls
    return cnt_class


@dataclass
class AlarmTracker:
    """
    Stateful trend alarm for a directory / video run. Call :meth:`step` once
    per processed frame (after NMS), in order.
    """

    class_names: Sequence[str]
    device: torch.device | str | None = None
    skip_alert_class_ids: frozenset[int] | None = None
    size_tol: int = 5
    size_last: int = 8
    rep_tol: float = 1.0
    rep_thr: int = 3
    rep_last: float = 5.0
    cnt_tol: int = 5
    cnt_last: int = 6

    def __post_init__(self) -> None:
        self._dev = torch.device(self.device) if self.device is not None else torch.device("cpu")
        self.names = list(self.class_names)
        self.nc = len(self.names)
        if self.skip_alert_class_ids is None:
            self.skip_alert_class_ids = frozenset({0})
        zf = torch.zeros(self.nc, 5, device=self._dev, dtype=torch.float32)
        zf[:, 2] = 1.0
        self.cnt_class = zf
        self.size_class = torch.zeros(self.nc, 4, device=self._dev, dtype=torch.float32)
        self.occ_point = torch.empty(0, 4, device=self._dev, dtype=torch.float32)
        self._frame = 0

    @property
    def frame_index(self) -> int:
        return self._frame

    @property
    def tensor_device(self) -> torch.device:
        return self._dev

    def step(self, xywh: torch.Tensor, class_ids: torch.Tensor) -> None:
        """
        ``xywh`` center-format [cx, cy, w, h] in pixel space (same as YOLO exports).
        ``class_ids`` length must match number of rows in ``xywh``.
        """
        frame_idx = self._frame
        self._frame += 1

        xywh = xywh.to(device=self._dev, dtype=torch.float32)
        class_ids = class_ids.to(device=self._dev, dtype=torch.long).reshape(-1)
        n = xywh.reshape(-1, 4).shape[0]
        if class_ids.numel() != n:
            raise ValueError("class_ids length must match number of boxes")

        img_ids = torch.full((n,), float(frame_idx), device=self._dev, dtype=torch.float32)

        if n == 0:
            zero_cls = torch.empty(0, dtype=torch.long, device=self._dev)
            img_one = torch.tensor([float(frame_idx)], device=self._dev, dtype=torch.float32)
            self.cnt_class = cnt(
                zero_cls,
                img_one,
                self.cnt_class,
                self.names,
                tol=self.cnt_tol,
                last_seen_window=self.cnt_last,
                skip_alert_class_ids=self.skip_alert_class_ids,
            )
            return

        self.size_class = size(
            xywh,
            class_ids,
            img_ids,
            self.size_class,
            self.names,
            tol=self.size_tol,
            last_seen_window=self.size_last,
            skip_alert_class_ids=self.skip_alert_class_ids,
        )
        self.occ_point = rep(img_ids, xywh, self.occ_point, tol=self.rep_tol, rep_thr=self.rep_thr, last=self.rep_last)
        self.cnt_class = cnt(
            class_ids,
            img_ids,
            self.cnt_class,
            self.names,
            tol=self.cnt_tol,
            last_seen_window=self.cnt_last,
            skip_alert_class_ids=self.skip_alert_class_ids,
        )


def configure_alarm_logging(*, quiet: bool = False) -> None:
    """Route alarm messages through logging (stderr) instead of bare prints."""
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.ERROR if quiet else logging.WARNING)
