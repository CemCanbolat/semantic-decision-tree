"""Curate tau2-bench retail tasks into SDT Examples.

Ground truth comes from evaluation_criteria.actions (the reference trajectory
that derives the target DB end state), never from model output. The mapping
from action space to decision labels lives in ACTION_PREFIX_MAP and is applied
by explicit rules below; every task lands in the audit CSV with its evidence,
and anything the rules cannot decide confidently is flagged ambiguous for the
hand audit the plan requires (~20 tasks before scaling).

tau2 retail facts this curator respects (verified against the pinned sources):
- No photo-proof flow, no chargeback concept: those labels never fire here.
- No 30-day return window: any delivered order may be returned on confirmation.
- Order IDs look like #W2378156; users are keyed by user_id like yusuf_rossi_9620.

Usage:
    python3.11 -m sdt.data.tau2_curate \
        --tau2-dir third_party/tau2-bench/data/tau2/domains/retail \
        --out-dir data/curated/tau2_retail
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sdt.data.loader import save_examples, save_json

from sdt.schemas import (
    ACTION_PREFIX_MAP,
    REFUND_INTENT_TOKENS,
    DecisionLabel,
    Example,
)

CROSS_DOMAIN_ACTION_MAPS = {
    "airline": {
        "book_reservation": DecisionLabel.BOOK_ORDER,
        "cancel_reservation": DecisionLabel.CANCEL_ORDER,
        "update_reservation_flights": DecisionLabel.MODIFY_ORDER,
        "update_reservation_baggages": DecisionLabel.MODIFY_ORDER,
        "update_reservation_passengers": DecisionLabel.MODIFY_ORDER,
        "transfer_to_human_agents": DecisionLabel.ESCALATE,
    },
    "telecom": {
        "grant_app_permission": DecisionLabel.GRANT_APP_PERMISSION,
        "set_network_mode_preference": DecisionLabel.SET_NETWORK_MODE,
        "toggle_airplane_mode": DecisionLabel.TOGGLE_AIRPLANE_MODE,
        "toggle_roaming": DecisionLabel.TOGGLE_ROAMING,
        "enable_roaming": DecisionLabel.TOGGLE_ROAMING,
        "toggle_data": DecisionLabel.TOGGLE_DATA,
        "toggle_data_saver_mode": DecisionLabel.TOGGLE_DATA_SAVER,
        "toggle_wifi_calling": DecisionLabel.TOGGLE_WIFI_CALLING,
        "refuel_data": DecisionLabel.REFUEL_DATA,
        "reboot_device": DecisionLabel.REBOOT_DEVICE,
        "reset_apn_settings": DecisionLabel.RESET_APN,
        "reseat_sim_card": DecisionLabel.RESEAT_SIM,
        "disconnect_vpn": DecisionLabel.DISCONNECT_VPN,
        "resume_line": DecisionLabel.RESUME_LINE,
        "make_payment": DecisionLabel.MAKE_PAYMENT,
        "send_payment_request": DecisionLabel.SEND_PAYMENT_REQUEST,
        "transfer_to_human_agents": DecisionLabel.ESCALATE,
    },
}


@dataclass
class Classification:
    label: DecisionLabel
    matched_action: Optional[str] = None
    mutating_actions: List[str] = field(default_factory=list)
    ambiguous: bool = False
    reasons: List[str] = field(default_factory=list)


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array (got %s)" % type(tasks).__name__)
    for t in tasks:
        for key in ("id", "user_scenario", "evaluation_criteria"):
            if key not in t:
                raise ValueError("task missing %r: %r" % (key, t.get("id")))
    return tasks


def _instructions(task: Dict[str, Any]) -> Dict[str, Any]:
    return task["user_scenario"]["instructions"]


def _actions(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return task["evaluation_criteria"].get("actions") or []


def _has_refund_intent(task: Dict[str, Any]) -> bool:
    ins = _instructions(task)
    blob = " ".join(
        str(ins.get(k) or "")
        for k in ("reason_for_call", "task_instructions", "known_info", "unknown_info")
    ).lower()
    return any(tok in blob for tok in REFUND_INTENT_TOKENS)


def classify_task(task: Dict[str, Any],
                  action_map: Optional[Dict[str, DecisionLabel]] = None) -> Classification:
    action_map = action_map if action_map is not None else ACTION_PREFIX_MAP
    actions = _actions(task)
    names = [a.get("name", "") for a in actions]

    if any(n.startswith("transfer_to_human_agents") for n in names):
        return Classification(
            label=DecisionLabel.ESCALATE,
            matched_action="transfer_to_human_agents",
            reasons=["transfer action in reference trajectory"],
        )

    mutating: List[Tuple[str, DecisionLabel]] = []
    for n in names:
        for prefix, label in action_map.items():
            if n.startswith(prefix) and label is not DecisionLabel.ESCALATE:
                mutating.append((n, label))
                break

    if mutating:
        distinct = sorted({label for _, label in mutating})
        terminal_name, terminal_label = mutating[-1]
        reasons: List[str] = []
        if len(distinct) > 1:
            reasons.append("multiple distinct mutating action categories: " + ", ".join(n for n, _ in mutating))
        return Classification(
            label=terminal_label,
            matched_action=terminal_name,
            mutating_actions=[n for n, _ in mutating],
            ambiguous=bool(reasons),
            reasons=reasons,
        )

    if _has_refund_intent(task):
        return Classification(
            label=DecisionLabel.POLITELY_DENY,
            ambiguous=True,
            reasons=[
                "no DB write + refund-ish intent: politely_deny vs needs_info "
                "cannot be separated without hand audit"
            ],
        )
    return Classification(
        label=DecisionLabel.OUT_OF_SCOPE,
        reasons=["no DB write and no refund-ish intent: info-only task"],
    )


def _collect_argument_values(task: Dict[str, Any], keys: Tuple[str, ...]) -> List[str]:
    vals: List[str] = []
    for a in _actions(task):
        args = a.get("arguments") or {}
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v:
                vals.append(v)
    return vals


def resolve_user(task: Dict[str, Any], db: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Resolve the task's user against db users. Returns (user_id, notes)."""
    notes: List[str] = []
    direct = _collect_argument_values(task, ("user_id",))
    users: Dict[str, Any] = db.get("users", {})
    for uid in direct:
        if uid in users:
            return uid, notes
    # name+zip / email resolution from find_* action arguments
    firsts = _collect_argument_values(task, ("first_name",))
    lasts = _collect_argument_values(task, ("last_name",))
    zips = _collect_argument_values(task, ("zip",))
    emails = _collect_argument_values(task, ("email",))
    for uid, u in users.items():
        name = u.get("name", {})
        if (firsts and lasts and name.get("first_name") in firsts and name.get("last_name") in lasts
                and (not zips or u.get("address", {}).get("zip") in zips)):
            return uid, ["resolved via name+zip"]
        if emails and u.get("email", "").lower() in [e.lower() for e in emails]:
            return uid, ["resolved via email"]
    notes.append("user unresolved: no direct id, name/zip, or email match")
    return None, notes


def resolve_orders(task: Dict[str, Any], db: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return {order_id: order_record} for order ids referenced by the task."""
    notes: List[str] = []
    referenced = list(dict.fromkeys(_collect_argument_values(task, ("order_id",))))
    # fallback: order ids mentioned only in prose ("order W4284542")
    ins = _instructions(task)
    blob = " ".join(str(ins.get(k) or "") for k in ("reason_for_call", "known_info"))
    import re
    for match in re.findall(r"#?W[0-9]{5,}", blob):
        cand = match if match.startswith("#") else "#" + match
        if cand not in referenced:
            referenced.append(cand)
    if not referenced:
        return {}, ["no order referenced by any action"]
    orders_db: Dict[str, Any] = db.get("orders", {}) or {}
    out: Dict[str, Any] = {}
    for oid in referenced:
        if oid in orders_db:
            out[oid] = orders_db[oid]
        else:
            stripped = oid.lstrip("#")
            if stripped in orders_db:
                out[oid] = orders_db[stripped]
            else:
                notes.append("order %s not found in db" % oid)
    return out, notes


def resolve_user_from_orders(orders: Dict[str, Any], db: Dict[str, Any]) -> Optional[str]:
    """Orders carry user_id: use it when the trajectory never names the user."""
    users = db.get("users", {})
    for order in orders.values():
        uid = order.get("user_id")
        if uid and uid in users:
            return uid
    return None


def resolve_products(task: Dict[str, Any], db: Dict[str, Any]) -> Dict[str, Any]:
    pids = _collect_argument_values(task, ("product_id",))
    products = db.get("products", {}) or {}
    return {p: products[p] for p in pids if p in products}


def _load_db(tau2_dir: Path) -> Dict[str, Any]:
    db_json = tau2_dir / "db.json"
    if db_json.exists():
        return json.loads(db_json.read_text(encoding="utf-8"))
    import tomllib
    return tomllib.loads((tau2_dir / "db.toml").read_text(encoding="utf-8"))


def _join_airline(task: Dict[str, Any], db: Dict[str, Any]) -> Tuple:
    users = db.get("users", {})
    uid = next((u for u in _collect_argument_values(task, ("user_id",)) if u in users), None)
    if uid is None:
        firsts = _collect_argument_values(task, ("first_name",))
        lasts = _collect_argument_values(task, ("last_name",))
        for u, rec in users.items():
            name = rec.get("name", {})
            if (firsts and lasts and name.get("first_name") in firsts
                    and name.get("last_name") in lasts):
                uid = u
                break
    reservations = {}
    for rid in _collect_argument_values(task, ("reservation_id",)):
        if rid in db.get("reservations", {}):
            reservations[rid] = db["reservations"][rid]
    if uid is None:
        for r in reservations.values():
            if r.get("user_id") in users:
                uid = r["user_id"]
                break
    flights = {p: db["flights"][p] for p in _collect_argument_values(task, ("flight_id",))
               if p in db.get("flights", {})}
    return uid, users.get(uid), reservations, flights, []


def _join_telecom(task: Dict[str, Any], db: Dict[str, Any]) -> Tuple:
    refs = _collect_argument_values(task, ("line_id", "device_id", "plan_id", "bill_id"))
    by_id: Dict[str, Dict[str, Any]] = {}
    for kind in ("lines", "devices", "plans", "bills"):
        for rec in db.get(kind, []):
            for key in ("line_id", "device_id", "plan_id", "bill_id"):
                if rec.get(key):
                    by_id[rec[key]] = rec
    entities = {r: by_id[r] for r in refs if r in by_id}
    customers = db.get("customers", [])
    customer = customers[0] if len(customers) == 1 else None
    return customer, None, entities, {}, []


def render_text(task: Dict[str, Any]) -> str:
    ins = _instructions(task)
    lines: List[str] = []
    rfc = (ins.get("reason_for_call") or "").strip()
    if rfc and rfc != ".":
        lines.append("Reason for contact: " + rfc)
    known = (ins.get("known_info") or "").strip()
    if known and known != ".":
        lines.append("Known: " + known)
    unknown = (ins.get("unknown_info") or "").strip()
    if unknown and unknown != ".":
        lines.append("Not remembering: " + unknown)
    ti = (ins.get("task_instructions") or "").strip()
    if ti and ti != ".":
        lines.append("Style note: " + ti)
    persona = (task["user_scenario"].get("persona") or "").strip()
    if persona and persona != ".":
        lines.append("Persona: " + persona)
    return chr(10).join(lines) if lines else "Reason for contact: (empty)"


def family_for(cls: Classification) -> str:
    if cls.matched_action:
        return "tau2:" + cls.matched_action
    if cls.label is DecisionLabel.POLITELY_DENY:
        return "tau2:no_write_refund_intent"
    return "tau2:no_write_info_only"


def curate(tau2_dir: Path, out_dir: Path, seed: int = 7,
           domain: str = "retail") -> Dict[str, Any]:
    tasks = load_tasks(tau2_dir / "tasks.json")
    db = _load_db(tau2_dir)
    action_map = CROSS_DOMAIN_ACTION_MAPS.get(domain, ACTION_PREFIX_MAP)
    splits_path = tau2_dir / "split_tasks.json"
    split_map: Dict[str, str] = {}
    if splits_path.exists():
        raw = json.loads(splits_path.read_text(encoding="utf-8"))
        for split, ids in raw.items():
            for tid in ids:
                split_map.setdefault(str(tid), split)

    examples: List[Example] = []
    audit_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for task in tasks:
        tid = str(task["id"])
        cls = classify_task(task, action_map)
        if domain == "airline":
            uid, user_record, orders, products, notes = _join_airline(task, db)
        elif domain == "telecom":
            user_record, _, orders, products, notes = _join_telecom(task, db)
            uid = None
        else:
            uid, user_notes = resolve_user(task, db)
            orders, order_notes = resolve_orders(task, db)
            if uid is None and orders:
                uid = resolve_user_from_orders(orders, db)
                if uid:
                    user_notes = user_notes + ["resolved via order.user_id"]
            user_record = db.get("users", {}).get(uid) if uid else None
            products = resolve_products(task, db)
            notes = user_notes + order_notes
        focus = next(iter(orders), None)

        context: Dict[str, Any] = {
            "user": user_record,
            "orders": orders,
            "focus_order_id": focus,
            "products": products,
        }
        if domain == "telecom":
            context["entities"] = orders
            context["orders"] = {}
            context["focus_order_id"] = None
            context["products"] = {}
        text = render_text(task)
        ex = Example(
            example_id="tau2_retail_" + tid,
            text=text,
            context=context,
            gold=cls.label,
            split=split_map.get(tid, "base"),
            family=family_for(cls),
            variant="verbatim",
            provenance={
                "source": "tau2-bench retail",
                "task_id": tid,
                "matched_action": cls.matched_action,
                "mutating_actions": cls.mutating_actions,
                "reward_basis": task["evaluation_criteria"].get("reward_basis"),
                "curation_notes": notes,
            },
        )
        examples.append(ex)
        audit_rows.append({
            "task_id": tid,
            "label": cls.label.value,
            "matched_action": cls.matched_action or "",
            "ambiguous": "yes" if cls.ambiguous else "no",
            "reasons": "; ".join(cls.reasons),
            "user_join": uid or "UNRESOLVED",
            "orders_joined": ",".join(orders.keys()),
            "reason_for_call": (_instructions(task).get("reason_for_call") or "")[:160].replace(chr(10), " "),
        })

    # guarantee the audit sample the plan requires, even on small inputs
    ambiguous_ids = {r["task_id"] for r in audit_rows if r["ambiguous"] == "yes"}
    confident = [r["task_id"] for r in audit_rows if r["ambiguous"] == "no"]
    need = max(0, 20 - len(ambiguous_ids))
    sample = sorted(ambiguous_ids | set(rng.sample(confident, min(need, len(confident)))))
    for row in audit_rows:
        row["audit_sample"] = "yes" if row["task_id"] in sample else "no"

    out_dir.mkdir(parents=True, exist_ok=True)
    save_examples(examples, str(out_dir / "examples.jsonl"))
    with (out_dir / "audit.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    label_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    join_failures = 0
    for ex in examples:
        label_counts[ex.gold.value] = label_counts.get(ex.gold.value, 0) + 1
        split_counts[ex.split] = split_counts.get(ex.split, 0) + 1
        family_counts[ex.family] = family_counts.get(ex.family, 0) + 1
        if ex.context["user"] is None:
            join_failures += 1
    card = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "tau2-bench retail",
        "tau2_dir": str(tau2_dir),
        "n_examples": len(examples),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "ambiguous_tasks": len(ambiguous_ids),
        "user_join_failures": join_failures,
        "order_join_failures": sum(1 for ex in examples if not ex.context["focus_order_id"]),
        "domain": domain,
        "taxonomy_note": (
            "Labels derive from reference-action DB writes (ACTION_PREFIX_MAP). "
            "request_photo_proof never fires on tau2 retail: the policy has no "
            "photo-proof flow. No 30-day return window exists in this policy."
        ),
        "seed": seed,
    }
    save_json(card, str(out_dir / "data_card.json"))
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau2-dir", type=Path, required=True,
                    help="directory containing tasks.json, db.json, split_tasks.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--domain", default="retail",
                    choices=["retail", "airline", "telecom"])
    ap.add_argument("--splits", default=None,
                    help="comma-separated split names to keep (base/train/test or full)")
    args = ap.parse_args()
    tau2_dir = args.tau2_dir
    if args.splits:
        import shutil, tempfile
        keep = set(args.splits.split(","))
        tasks_all = load_tasks(tau2_dir / "tasks.json")
        raw = json.loads((tau2_dir / "split_tasks.json").read_text(encoding="utf-8"))
        split_map = {}
        for split, ids in raw.items():
            for tid in ids:
                split_map.setdefault(str(tid), set()).add(split)
        wanted = {str(t["id"]) for t in tasks_all if keep & split_map.get(str(t["id"]), set())}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copytree(tau2_dir, tmp_dir, dirs_exist_ok=True)
            (tmp_dir / "tasks.json").write_text(
                json.dumps([t for t in tasks_all if str(t["id"]) in wanted]), encoding="utf-8")
            card = curate(tmp_dir, args.out_dir, args.seed, args.domain)
    else:
        card = curate(tau2_dir, args.out_dir, args.seed, args.domain)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
