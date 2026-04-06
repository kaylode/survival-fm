
def evaluate_subgroups(
    df_test,
    duration_col: str,
    event_col: str,
    risk_scores: np.ndarray,
    surv_probs: Optional[np.ndarray],
    surv_times: Optional[np.ndarray],
    subgroup_col: str,
    bins: Optional[list] = None,
    labels: Optional[list] = None,
    df_train=None,
) -> dict:
    """Compute C-index (and optionally IBS) per subgroup of the test set.

    Parameters
    ----------
    df_test        : test-set DataFrame with feature + outcome columns.
    duration_col   : name of time column.
    event_col      : name of event column.
    risk_scores    : (n_test,) risk array.
    surv_probs     : (n_test, T) survival matrix, or None.
    surv_times     : (T,) time array, or None.
    subgroup_col   : column name to stratify by.
    bins           : bin edges for numeric columns (passed to pd.cut); if None
                     and the column is numeric, quartiles are used.
    labels         : bin labels; defaults to string ranges.
    df_train       : training DataFrame (needed for IBS IPCW weights);
                     if None, IBS is skipped.

    Returns
    -------
    dict mapping subgroup label → metrics dict (C-index, optional IBS).
    """
    col = df_test[subgroup_col] if subgroup_col in df_test.columns else None
    if col is None:
        warnings.warn(f"Column '{subgroup_col}' not in df_test; skipping subgroup eval.")
        return {}

    if bins is not None or pd.api.types.is_numeric_dtype(col):
        if bins is None:
            bins = list(np.percentile(col.dropna(), [0, 25, 50, 75, 100]))
        groups = pd.cut(col, bins=bins, labels=labels, include_lowest=True)
    else:
        groups = col

    results = {}
    for grp_label, grp_mask in groups.groupby(groups):
        idx = grp_mask.index.tolist()
        if len(idx) < 5:
            continue

        sub_test        = df_test.loc[idx].reset_index(drop=True)
        sub_risk        = risk_scores[idx]
        sub_surv        = surv_probs[idx] if surv_probs is not None else None

        try:
            c_idx, _, _, _, _ = concordance_index_censored(
                sub_test[event_col].astype(bool).values,
                sub_test[duration_col].values,
                sub_risk,
            )
        except Exception:
            c_idx = float("nan")

        m = {"C-index": c_idx, "n": len(idx)}

        if df_train is not None and sub_surv is not None and surv_times is not None:
            try:
                from survpfn.metrics.single import evaluate_sr
                m["IBS"] = evaluate_sr(
                    df_train, sub_test, duration_col, event_col,
                    sub_risk, surv_probs=sub_surv, surv_times=surv_times,
                ).get("IBS", float("nan"))
            except Exception:
                m["IBS"] = float("nan")

        results[str(grp_label)] = m

    return results


def plot_calibration_curve(
    y_test,
    duration_col: str,
    event_col: str,
    surv_probs: np.ndarray,
    surv_times: np.ndarray,
    time_horizon: float,
    n_bins: int = 10,
    ax=None,
    label: str = "",
):
    """Plot observed vs predicted event probability at a fixed time horizon.

    Parameters
    ----------
    y_test         : test DataFrame with outcome columns.
    duration_col   : time column name.
    event_col      : event column name.
    surv_probs     : (n_test, T) survival matrix.
    surv_times     : (T,) time points.
    time_horizon   : fixed time at which to evaluate.
    n_bins         : number of calibration bins.
    ax             : matplotlib axis; if None a new figure is created.
    label          : legend label for this curve.

    Returns
    -------
    ax : the matplotlib axis with the calibration curve plotted.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib required: uv add matplotlib") from e

    # Predicted probability of event by time_horizon = 1 - S(t*)
    pred_event = 1.0 - np.array([
        float(np.interp(time_horizon, surv_times, surv_probs[i]))
        for i in range(len(surv_probs))
    ])

    T = y_test[duration_col].values
    E = y_test[event_col].values.astype(bool)

    # Observed: did event occur by time_horizon? (ignoring censoring before t*)
    observed = ((T <= time_horizon) & E).astype(float)

    # Bin by predicted probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx   = np.digitize(pred_event, bin_edges[1:-1])
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(pred_event[mask].mean())
        mean_obs.append(observed[mask].mean())
        counts.append(mask.sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(mean_pred, mean_obs, marker="o", label=label or "model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel(f"Mean predicted P(event ≤ {time_horizon:.1f})")
    ax.set_ylabel("Observed fraction")
    ax.set_title(f"Calibration curve (t={time_horizon:.1f})")
    ax.legend()
    return ax

