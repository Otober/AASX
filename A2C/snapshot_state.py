
import pandas as pd, numpy as np, re, math
from typing import Dict, Any, Tuple, Optional, List

EVENT_QUEUED   = "queued"
EVENT_START    = "start"
EVENT_END      = "end"
EVENT_FETCH    = "fetch"
EVENT_DELIVERY = "delivery"
EVENT_DONE     = "done"

MACHINE_IDS = [f"M{i}" for i in range(1, 9)]
AGV_IDS     = [str(i) for i in range(1, 6)]  # 'part' col is numeric string for AGVs during move events
JOB_IDS     = [f"J{i}" for i in range(1, 41)]

def _parse_edge(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse strings like 'M1->GEN', 'GEN->M3', or 'M4' into (src, dst).
    Returns (None, None) if no arrow exists and s is a single node.
    """
    if not isinstance(s, str) or not s:
        return (None, None)
    if "->" in s:
        a, b = s.split("->", 1)
        return (a, b)
    return (s, None)

def _is_agv_id(val) -> bool:
    try:
        s = str(val)
        # AGV ids appear as integers in 'part' when event is fetch/delivery in this log
        return s.isdigit() and s in { "1","2","3","4","5" }
    except Exception:
        return False

class TraceSnapshot:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Normalize column names and types
        self.df.columns = [c.strip() for c in self.df.columns]
        # time column
        if "time" not in self.df.columns:
            raise ValueError("Expected a 'time' column in the trace.")
        self.df["time"] = pd.to_numeric(self.df["time"], errors="coerce")
        # clean strings
        for c in ["part","job","operation","machine","event"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str)
        # convenience
        self.df.sort_values(["time"], inplace=True, kind="mergesort")
        self.df.reset_index(drop=True, inplace=True)

    def _ops_by_machine_at_t(self, t: float) -> Dict[str, Dict[str, Any]]:
        """
        For each machine:
          - sum_pt_on_queue: sum of nominal processing times for ops waiting in input queue at time t
          - processing_remaining: remaining time for the op currently processing (if any)
          - outq_waiting_count: # of ops finished (END before t) but not yet fetched (FETCH after t or none)
        """
        out: Dict[str, Dict[str, Any]] = {m: {"sum_pt_on_queue": 0.0,
                                              "processing_remaining": 0.0,
                                              "outq_waiting_count": 0} for m in MACHINE_IDS}
        # For efficiency, precompute start/end times per (job,operation,machine)
        gcols = ["job","operation","machine"]
        # Identify first start and first end for each op@machine
        starts = (self.df[self.df["event"]==EVENT_START]
                    .groupby(gcols)["time"].min().rename("t_start"))
        ends   = (self.df[self.df["event"]==EVENT_END]
                    .groupby(gcols)["time"].min().rename("t_end"))
        queued = (self.df[self.df["event"]==EVENT_QUEUED]
                    .groupby(gcols)["time"].min().rename("t_queued"))
        # FETCH events are logged with machine like 'M1->GEN' and event 'fetch'
        fetches = self.df[self.df["event"]==EVENT_FETCH].copy()
        # Map fetch event back to source machine
        fetches["src"], fetches["dst"] = zip(*fetches["machine"].map(lambda s: s.split("->",1) if "->" in s else (s, None)))
        # For each op, mark if it has been fetched yet (t_fetch)
        # We don't know the op id at FETCH rows; we approximate by matching same job+operation and source machine.
        # Create a left-join-able key for machine source only rows in ops space.
        # First, put together a frame of all ops with their machines:
        # NOTE: an op is tied to one machine row by the log itself.
        base = (self.df[gcols].drop_duplicates().set_index(gcols))
        base = base.join(queued, how="left").join(starts, how="left").join(ends, how="left")
        base = base.reset_index()
        # Build a lookup for (job, operation, src_machine) -> earliest fetch time
        fkey = fetches.assign(src_machine=fetches["src"]).groupby(["job","operation","src_machine"])["time"].min().rename("t_fetch")
        # Merge fetch onto base
        base = base.merge(fkey.reset_index(), how="left",
                          left_on=["job","operation","machine"], right_on=["job","operation","src_machine"])
        # For each machine, compute the stats
        for m in MACHINE_IDS:
            sub = base[base["machine"]==m]
            if sub.empty:
                continue
            # 1) Sum of processing times for ops currently waiting in input queue at time t
            # Waiting in input queue: queued <= t, start > t or start is NaN
            wait_mask = (sub["t_queued"].notna()) & (sub["t_queued"] <= t) & (sub["t_start"].isna() | (sub["t_start"] > t))
            wait_ops = sub[wait_mask]
            # processing time for each op = t_end - t_start when both exist later
            # If t_end is missing but t_start exists later, we'll attempt to estimate from global median duration for that machine
            durations_all = (sub[ sub["t_start"].notna() & sub["t_end"].notna() ]["t_end"] - sub[ sub["t_start"].notna() & sub["t_end"].notna() ]["t_start"]).dropna()
            med_pt = float(durations_all.median()) if len(durations_all) else 0.0
            pts = []
            for _, r in wait_ops.iterrows():
                if not math.isnan(r.get("t_start", math.nan)) and not math.isnan(r.get("t_end", math.nan)):
                    pts.append(float(r["t_end"] - r["t_start"]))
                elif not math.isnan(r.get("t_start", math.nan)) and math.isnan(r.get("t_end", math.nan)):
                    # has a start somewhere but no end -> fallback to median
                    pts.append(med_pt)
                else:
                    # no start known yet; use machine median
                    pts.append(med_pt)
            out[m]["sum_pt_on_queue"] = float(sum(pts)) if pts else 0.0

            # 2) Remaining processing time for the op currently in process (if any)
            # In process: start <= t and (end is NaN or end > t)
            proc_mask = (sub["t_start"].notna()) & (sub["t_start"] <= t) & ( (sub["t_end"].isna()) | (sub["t_end"] > t) )
            # If multiple such rows (shouldn't happen), pick the one with the latest t_start
            inproc = sub[proc_mask].copy()
            if not inproc.empty:
                r = inproc.sort_values("t_start").iloc[-1]
                # Remaining = (t_end - t) if t_end exists; else estimate using median PT minus elapsed
                if not math.isnan(r.get("t_end", math.nan)):
                    out[m]["processing_remaining"] = float(max(0.0, r["t_end"] - t))
                else:
                    elapsed = float(max(0.0, t - r["t_start"]))
                    out[m]["processing_remaining"] = float(max(0.0, med_pt - elapsed))
            else:
                out[m]["processing_remaining"] = 0.0

            # 3) Count of ops finished but not yet fetched (waiting at output)
            outq_mask = (sub["t_end"].notna()) & (sub["t_end"] <= t) & ( (sub["t_fetch"].isna()) | (sub["t_fetch"] > t) )
            out[m]["outq_waiting_count"] = int(outq_mask.sum())

        return out

    def _agv_state_at_t(self, t: float) -> Dict[str, Dict[str, Any]]:
        """
        For each AGV (1..5):
          - status in {"delivery", "fetch", "idle"}
          - location: most recent location token (M1, GEN, or edge M1->GEN / GEN->M3) known by or before t
        """
        agv_state = {str(i): {"status": "idle", "location": None} for i in range(1,6)}
        # Consider only rows that are AGV-related: event in {fetch, delivery, done} and part is an AGV id
        is_agv = self.df["part"].apply(lambda v: str(v).isdigit())
        df_agv = self.df[ is_agv & self.df["event"].isin([EVENT_FETCH, EVENT_DELIVERY, EVENT_DONE]) ].copy()
        if df_agv.empty:
            return agv_state

        for agv_id, rows in df_agv.groupby("part"):
            # Rows up to time t
            past = rows[ rows["time"] <= t ].sort_values("time")
            if past.empty:
                continue
            last = past.iloc[-1]
            ev   = last["event"]
            mach = last["machine"]
            src, dst = _parse_edge(mach)
            if ev == EVENT_DONE:
                agv_state[str(agv_id)]["status"] = "idle"
                # location stays where it last was (we'll derive from the last non-done row if any)
                prev = past[past["event"].isin([EVENT_FETCH, EVENT_DELIVERY])]
                if not prev.empty:
                    p = prev.iloc[-1]
                    s, d = _parse_edge(p["machine"])
                    # If last move was X->Y and it's 'delivery', assume arrived at Y
                    if p["event"] == EVENT_DELIVERY and d:
                        agv_state[str(agv_id)]["location"] = d
                    elif p["event"] == EVENT_FETCH and s:
                        agv_state[str(agv_id)]["location"] = s
            elif ev in (EVENT_FETCH, EVENT_DELIVERY):
                agv_state[str(agv_id)]["status"] = ev
                # For location, use source for fetch, dest for delivery if available
                if ev == EVENT_FETCH:
                    agv_state[str(agv_id)]["location"] = src or mach
                else:
                    agv_state[str(agv_id)]["location"] = dst or mach

        # Fill missing locations with 'GEN' as a default hub if nothing known
        for k, v in agv_state.items():
            if v["location"] in (None, "", "nan"):
                v["location"] = "GEN"
        return agv_state

    def _jobs_state_at_t(self, t: float) -> Dict[str, Dict[str, Any]]:
        """
        For each job J1..J40:
          - current_operation: last seen 'operation' by time t (or None)
          - status in {"GEN", "waiting_process", "processing", "waiting_delivery"}
        """
        out: Dict[str, Dict[str, Any]] = {j: {"current_operation": None, "status": "GEN"} for j in JOB_IDS}
        # We will analyze per (job, operation) regardless of machine (each op appears on one machine in log)
        gcols = ["job","operation"]
        # Map op -> first queued, start, end, fetch, delivery times (taking mins)
        queued = (self.df[self.df["event"]==EVENT_QUEUED].groupby(gcols)["time"].min().rename("t_queued"))
        starts = (self.df[self.df["event"]==EVENT_START ].groupby(gcols)["time"].min().rename("t_start"))
        ends   = (self.df[self.df["event"]==EVENT_END   ].groupby(gcols)["time"].min().rename("t_end"))
        # FETCH/DELIVERY tracked with machine edges but still have job+operation
        fetches = (self.df[self.df["event"]==EVENT_FETCH].groupby(gcols)["time"].min().rename("t_fetch"))
        delivs  = (self.df[self.df["event"]==EVENT_DELIVERY].groupby(gcols)["time"].min().rename("t_delivery"))

        ops = (self.df[["job","operation"]].drop_duplicates()
                    .merge(queued, how="left", on=gcols)
                    .merge(starts, how="left", on=gcols)
                    .merge(ends,   how="left", on=gcols)
                    .merge(fetches,how="left", on=gcols)
                    .merge(delivs, how="left", on=gcols)
               )

        # For each job, determine the latest operation "reached" by time t
        for job, jops in ops.groupby("job", sort=False):
            # reached if any of these times <= t
            reached = jops[
                (jops[["t_queued","t_start","t_end","t_fetch","t_delivery"]].fillna(float("inf")).min(axis=1) <= t)
            ].copy()
            if reached.empty:
                # job has not entered the system yet
                out[job] = {"current_operation": None, "status": "GEN"}
                continue
            # Pick the operation with the largest "stage index" at time t; we approximate stage order by the tuple of event times
            # We compute a progress score: delivery < fetch < end < start < queued in terms of which is latest <= t
            def _stage_at_row(r):
                times = {
                    "delivery": r.get("t_delivery", math.inf),
                    "fetch":    r.get("t_fetch",    math.inf),
                    "end":      r.get("t_end",      math.inf),
                    "start":    r.get("t_start",    math.inf),
                    "queued":   r.get("t_queued",   math.inf),
                }
                # Identify the last event that occurred by time t
                occur = {k:v for k,v in times.items() if v <= t}
                if not occur:
                    return ("GEN", -1.0)
                last_ev = max(occur.items(), key=lambda kv: kv[1])[0]
                order = {"queued":1, "start":2, "end":3, "fetch":4, "delivery":5}
                return (last_ev, order[last_ev])

            # Determine the op with max stage order
            reached["_stage"] = reached.apply(lambda r: _stage_at_row(r), axis=1, result_type="reduce")
            reached["_order"] = reached["_stage"].apply(lambda x: x[1])
            best = reached.sort_values(["_order", "t_queued","t_start","t_end","t_fetch","t_delivery"], na_position="last").iloc[-1]
            current_op = str(best["operation"])
            last_event = best["_stage"][0]

            # Translate to user-facing status
            if last_event == "queued":
                status = "waiting process in input queue"
            elif last_event == "start":
                status = "processing"
            elif last_event in ("end", "fetch"):
                # If ended but no fetch/delivery by t, it's waiting for delivery
                # If fetch happened by t but delivery not yet, still waiting
                if ( (pd.notna(best.get("t_end")) and best["t_end"] <= t) and
                     ( (pd.isna(best.get("t_fetch")) or best["t_fetch"] > t) or
                       (pd.isna(best.get("t_delivery")) or best["t_delivery"] > t) ) ):
                    status = "waiting delivery in output queue"
                else:
                    status = "waiting delivery in output queue"
            elif last_event == "delivery":
                # After delivery, the next op (if any) may exist; but as of exactly time t, this op is delivered.
                status = "GEN"  # delivered back to GEN (buffer) before next routing
            else:
                status = "GEN"

            out[job] = {"current_operation": current_op, "status": status}

        # Ensure all declared jobs exist even if unseen
        for jid in [f"J{i}" for i in range(1,41)]:
            out.setdefault(jid, {"current_operation": None, "status": "GEN"})

        return out

    def snapshot(self, t: float) -> Dict[str, Any]:
        return {
            "t": float(t),
            "machines": self._ops_by_machine_at_t(t),
            "agvs": self._agv_state_at_t(t),
            "jobs": self._jobs_state_at_t(t),
        }

    def _utilization_until_t(self, t: float) -> float:
        if t <= 0:
            return 0.0
        eps = 1e-9
        gcols = ["job","operation","machine"]
        starts = (self.df[self.df["event"]=="start"].groupby(gcols)["time"].min().rename("t_start"))
        ends   = (self.df[self.df["event"]=="end"  ].groupby(gcols)["time"].min().rename("t_end"))
        base = (self.df[gcols].drop_duplicates().set_index(gcols))
        base = base.join(starts, how="left").join(ends, how="left").reset_index()

        util_vals = []
        for m in [f"M{i}" for i in range(1,9)]:
            sub = base[base["machine"]==m].copy()
            if sub.empty:
                util_vals.append(0.0)
                continue
            mask_done = sub["t_start"].notna() & sub["t_end"].notna()
            med_pt = float((sub.loc[mask_done, "t_end"] - sub.loc[mask_done, "t_start"]).median()) if mask_done.any() else 0.0
            busy = 0.0
            for _, r in sub.iterrows():
                ts = r.get("t_start", float("nan"))
                te = r.get("t_end",   float("nan"))
                if pd.isna(ts):
                    continue
                if pd.isna(te):
                    te = ts + med_pt
                left  = max(0.0, float(ts))
                right = min(float(t), float(te))
                busy += max(0.0, right - left)
            util_vals.append( busy / max(float(t), eps) )
        return float(np.mean(util_vals)) if util_vals else 0.0

    def _workload_variance_and_qpending(self, t: float) -> tuple[float, int]:
        gcols = ["job","operation","machine"]
        starts = (self.df[self.df["event"]=="start" ].groupby(gcols)["time"].min().rename("t_start"))
        ends   = (self.df[self.df["event"]=="end"   ].groupby(gcols)["time"].min().rename("t_end"))
        queued = (self.df[self.df["event"]=="queued"].groupby(gcols)["time"].min().rename("t_queued"))
        fetches = self.df[self.df["event"]=="fetch"].copy()
        fetches["src"], fetches["dst"] = zip(*fetches["machine"].map(lambda s: s.split("->",1) if "->" in s else (s, None)))
        base = (self.df[gcols].drop_duplicates().set_index(gcols))
        base = base.join(queued, how="left").join(starts, how="left").join(ends, how="left").reset_index()
        fkey = fetches.assign(src_machine=fetches["src"]).groupby(["job","operation","src_machine"])["time"].min().rename("t_fetch")
        base = base.merge(fkey.reset_index(), how="left",
                          left_on=["job","operation","machine"], right_on=["job","operation","src_machine"])

        workloads = []
        qpending = 0
        for m in [f"M{i}" for i in range(1,9)]:
            sub = base[base["machine"]==m]
            if sub.empty:
                workloads.append(0)
                continue
            inq_mask = (sub["t_queued"].notna()) & (sub["t_queued"] <= t) & (sub["t_start"].isna() | (sub["t_start"] > t))
            inq_count = int(inq_mask.sum())
            proc_mask = (sub["t_start"].notna()) & (sub["t_start"] <= t) & ( (sub["t_end"].isna()) | (sub["t_end"] > t) )
            proc_count = int(proc_mask.sum())
            outq_mask = (sub["t_end"].notna()) & (sub["t_end"] <= t) & ( (sub["t_fetch"].isna()) | (sub["t_fetch"] > t) )
            outq_count = int(outq_mask.sum())

            workload = inq_count + (1 if proc_count>0 else 0) + outq_count
            workloads.append(workload)
            qpending += inq_count

        sigma_var = float(np.var(workloads)) if workloads else 0.0
        return sigma_var, int(qpending)

    # Enhanced snapshot that adds rho_util, sigma_var, Qpending
    def snapshot(self, t: float) -> Dict[str, Any]:
        base = {
            "t": float(t),
            "machines": self._ops_by_machine_at_t(t),
            "agvs": self._agv_state_at_t(t),
            "jobs": self._jobs_state_at_t(t),
        }
        base["rho_util"] = self._utilization_until_t(t)
        sigma_var, qpending = self._workload_variance_and_qpending(t)
        base["sigma_var"] = sigma_var
        base["Qpending"]  = qpending
        return base

def load_trace(path: str) -> "TraceSnapshot":
    df = pd.read_csv(path)
    return TraceSnapshot(df)

