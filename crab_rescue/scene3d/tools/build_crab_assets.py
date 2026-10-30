"""Convert the Jumper (crab) URDF + STL meshes into web assets for the 3D rescue scene.

    python build_crab_assets.py --urdf <.../assembled/jumper/robot.urdf> --out ../assets

Writes crab.glb (one decimated mesh per link, flat, named after the link) and crab_kin.json
(the kinematic tree: joint origins, axes, limits, link colours, the display_home pose).
The robot's own geometry is used unchanged apart from decimation; no second crab is made.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

HOME = {  # preview-pose.json / scene.xml "display_home" of the assembled model
    "LF_J0_joint": -0.5507, "LF_J1_joint": -1.0658, "LF_J2_joint": -0.4408, "LF_J3_joint": -1.4226,
    "LF_J4_joint": 0.0, "RF_J0_joint": 0.5524, "RF_J1_joint": -1.066, "RF_J2_joint": 0.4414,
    "RF_J3_joint": 1.4212, "RF_J4_joint": 0.1, "LM_J0_joint": 0.0012, "LM_J1_joint": 0.5356,
    "LM_J2_joint": -1.8321, "RM_J0_joint": -0.0012, "RM_J1_joint": -0.5376, "RM_J2_joint": 1.8364,
    "LR_J0_joint": 0.6313, "LR_J1_joint": 0.5534, "LR_J2_joint": -1.8708, "RR_J0_joint": -0.6326,
    "RR_J1_joint": -0.5548, "RR_J2_joint": 1.8739}
BASE_HEIGHT = 0.10704


def floats(s: str | None, n: int, default: float = 0.0) -> list[float]:
    return [float(v) for v in s.split()] if s else [default] * n


def target_faces(n: int) -> int:
    return 14000 if n > 40000 else max(1500, min(n, 4500))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    urdf = Path(a.urdf)
    root = ET.parse(urdf).getroot()
    mats = {m.get("name"): [float(v) for v in m.find("color").get("rgba").split()]
            for m in root.findall("material") if m.find("color") is not None}
    scene = trimesh.Scene()
    links: dict[str, dict] = {}
    for link in root.findall("link"):
        name = link.get("name")
        pick = None
        for v in link.findall("visual"):          # skip the grey duplicate "collision_visual"
            if "collision_visual" in v.get("name", ""):
                continue
            pick = v
            break
        if pick is None:
            links[name] = {"mesh": None}
            continue
        mesh_path = urdf.parent / pick.find("geometry/mesh").get("filename")
        m = trimesh.load_mesh(mesh_path, process=True)
        m.merge_vertices()
        full = len(m.faces)
        if full > target_faces(full):
            m = m.simplify_quadric_decimation(face_count=target_faces(full))
        rgba = mats[pick.find("material").get("name")]
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[int(c * 255) for c in rgba[:3]] + [255],
            metallicFactor=0.15, roughnessFactor=0.55, name=name))
        org = pick.find("origin")
        xyz, rpy = floats(org.get("xyz"), 3), floats(org.get("rpy"), 3)
        scene.add_geometry(m, node_name=name, geom_name=name)
        links[name] = {"mesh": name, "color": rgba, "visual_xyz": xyz, "visual_rpy": rpy,
                       "faces_full": full, "faces": len(m.faces)}
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        ax = j.find("axis")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": floats(o.get("xyz"), 3), "rpy": floats(o.get("rpy"), 3),
            "axis": floats(ax.get("xyz"), 3) if ax is not None else None,
            "lower": float(lim.get("lower")) if lim is not None else None,
            "upper": float(lim.get("upper")) if lim is not None else None})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene.export(out / "crab.glb")
    (out / "crab_kin.json").write_text(json.dumps(
        {"source": "jumper-original assembled robot.urdf (V1.6-1)", "root": "base_link",
         "base_height_m": BASE_HEIGHT, "home": HOME, "links": links, "joints": joints},
        indent=1))
    kb = (out / "crab.glb").stat().st_size / 1024
    print(f"crab.glb {kb:.0f} KiB, {sum(v.get('faces', 0) for v in links.values())} faces "
          f"(from {sum(v.get('faces_full', 0) for v in links.values())}), {len(joints)} joints")


if __name__ == "__main__":
    main()
