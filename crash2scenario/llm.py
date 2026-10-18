"""Three-stage LLM scenario design used in the paper.

1. Paraphrase crash narratives into one simulator-feasible narrative.
2. Turn that narrative into a MetaDrive config dict.
3. Fine-tune numeric fields using logged minimum gap.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

VALID_MODELS = (
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
)

EXAMPLE_CONFIG = {
    "map": "X",
    "ego_turn": "left",
    "lead_turn": "straight",
    "traffic_density": 0.08,
    "accident_prob": 0.0,
    "steps": 320,
    "render": {"screen_size": [700, 900], "camera_position": [60, -54]},
    "vehicles": [
        {
            "id": "lead_braker",
            "lane_offset": 0,
            "delta_s_m": 18.0,
            "delta_t_m": 0.0,
            "direction": "same",
            "policy": "sudden_brake",
            "params": {
                "target_speed": 22.0,
                "brake_step": 55,
                "brake_steps": 120,
                "brake_mag": 1.0,
                "speed_kp": 0.08,
                "lookahead_s": 6.0,
            },
        }
    ],
    "cones": [],
}


def _client(model_name: str):
    if model_name not in VALID_MODELS:
        raise ValueError(f"Model not recognized: {model_name}. Choose one of {VALID_MODELS}.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key, "
            "or export OPENAI_API_KEY."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def call_llm(prompt: str, model_name: str = "gpt-5.2", temperature: float = 0.2, max_tokens: int = 5000) -> str:
    """Query the model and return the text response."""
    client = _client(model_name)
    if model_name == "o3" and temperature != 1.0:
        temperature = 1.0
    messages = [{"role": "user", "content": prompt}]
    if model_name in ("gpt-5.2", "gpt-5-mini"):
        response = client.responses.create(
            model=model_name,
            input=messages,
            max_output_tokens=max_tokens,
            reasoning={"effort": "low"},
        )
        return response.output_text
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a model response that should contain one JSON object."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(s[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Scenario response must be a JSON object.")
    return parsed


def paraphrase_narrative(narrative_list, scene: Dict[str, Any], model_name: str = "gpt-5.2") -> str:
    pre_crash_cp = scene["CP_PreCrash"]
    pre_crash_sv = scene["SV_PreCrash"]
    work_zone = scene["Work_Zone"]
    road_type = scene["Road_Type"]
    prompt = f"""
    Shown below are failure scenarios from actual historical crash records.
    Use these as examples of realistic crashes and propose a new scenario that represents a scenario with the following attributes:
        Pre-Crash movement (Crash Partner): {pre_crash_cp}
        Pre-Crash movement (Self Vehicle): {pre_crash_sv}
        Road Type: {road_type}
        Work zone: {work_zone}
    Narrative lists:{narrative_list}
    Do not design scenarios that are rear-end collisions for ego vehicle, or where the ego vehicle is being hit on by a non-ego vehicle from behind, because we would like to test the ego vehicle's policy at braking at a safe distance.
    Also do not design scenarios where there is head-on collision between ego and non-ego vehicle coming at each other from opposite directions.
    For example, avoid narratives such as "An oncoming passenger vehicle in the opposite direction entered the intersection and initiated a left turn across the Zoox vehicle’s path, appearing to misjudge the Zoox vehicle’s speed and gap."
    We have capability to control spawn positions of ego and non ego vehicles, scatter cones, control target speeds of each vehicle and control movement of non-ego vehicle as sudden braking or accelerate then brake, or nominal path following.
    So design a scenario that can be recreated using these variables, do not add additional details. At present, sharp changes in trajectory curvature, direction of motion of vehicles cannot be adjusted.
    Output format: Return a natural language output (text/str) that mimics a narrative.
    """
    return call_llm(prompt, model_name=model_name, max_tokens=800)


def scenario_prompt(narrative: str, dict_examples: Optional[dict] = None) -> str:
    examples = dict_examples if dict_examples is not None else EXAMPLE_CONFIG
    return f"""
    Generate a Metadrive scenario template in the form of a json using the dict examples shown below. The goal of scenario creation is to generate potential failure scenarios for ego vehicle, alongwith a Non-ego Crash Partener (NCP).
    The policy being used by the ego agent is IDM policy, which is designed to avoid. Hence, nominal failures such as naive crashes are difficult to observe with IDM. Create simple cases following the description of crash scenario, preferably using one NCP or more NCPs.
    These can be static objects as well, such as traffic cones.

    The scenario must adhere to the following description:
    Crash scenario desired: {narrative}

    Examples of a dict used to generate testing scenarios in Metadrive:
    {examples}
    Instructions:
    Choose map type carefully based on crash scenario description. You can choose 'X', 'O', 'T', 'S', 'C' for intersection, circle, three way intersection, straight and curved highway/street respectively. If the description simply mentions highway/street, choose between 'S' or 'C'.
    Avoid silly scenarios such as situating obstacles at the beginning of the trajctory causing a crash at T=0.
    Each non ego vehicle is initialized relative to ego (same lane coords). Following are the permissible entries for NCP description:
    For the ego and lead vehicle, specify the turn direction, as 'left','right' or 'straight' as:
    ego_turn: 'left'|'right'|'straight'
    lead_turn: 'left'|'right'|'straight'
    Each item:
    - id: string
    - direction: "same" | "opposite" (dictates direction of ego with respect to closest non ego vehicle)
    - lane_offset: 0=same as ego lane, +/-1 adjacent
    - delta_s_m: longitudinal offset ahead (negative behind)
    - delta_t_m: lateral offset within lane (meters)
    - policy: "sudden_brake" | "accel_then_brake" | "idm_nominal"
    - params: dict passed to that policy

    Here, sudden_brake policy brakes NCP suddenly, with the following params dict entries:
    - target_speed: target speed of NCP
    - brake_step: timestep at which NCP brakes
    - brake_steps: duration for braking
    - speed_kp: controls the acceleration when not braking as accel = speed_kp * (target_speed - v). If unsure, do not provide anything.

    For accel_then_brake policy, the vehicle accelerates, brakes and then follows a constant cruise speed. The param dict entries are:
    - accel_mag: magnitude of acceleration at the acceleration phase
    - accel_steps: t<accel_steps vehicle accelerates with accel_mag
    - cruise_speed: after braking, target speed
    - brake_step: step at which vehicle brakes
    - brake_steps: duration for brake starting brake_step
    - speed_kp: after braking, accel = speed_kp * (self.cruise_speed - v). If unsure, do not provide anything.

    For idm_nominal, NCP uses IDM Policy to follow a nominal speed with param with the entry:
    - target_speed_mps

    Stick to the format and entries of the dict using the example that can be used directly with Metadrive, do not add irrelevant details. Give a very simple implementation.
    Only use cones if the work zone indicates 'Yes', or if the narrative provides information that other static obstacles or work zone might have been present on-site.
    Make sure all entries are valid. For example, avoid having incomplete dicts leading to errors such as "Can't find valid lane for navigation."
    Return JSON only.
    """


def design_config(narrative: str, model_name: str = "gpt-5.2", dict_examples: Optional[dict] = None) -> Dict[str, Any]:
    raw = call_llm(scenario_prompt(narrative, dict_examples), model_name=model_name, max_tokens=5000)
    return parse_json_response(raw)


def fine_tune(config: dict, metrics: dict, narrative: str, model_name: str = "gpt-5.2") -> Dict[str, Any]:
    prompt = f"""
    You have to fine-tune the design parameters of a Metadrive scenario given in the form of a dict. The scenario was played with the provided dict and some logged metrics for this scenario will be provided.
    You are being asked to optimize the scenario parameters because a crash or collision or accident was expected, but it didnt happen. Analyze the scenario dict, recreate the scenario spatially and think what changes can be made to parameters such as speed, positioning etc to induce a collision/crash/accident/unexpected behavior such as abrupt braking, oscillation, etc.
    Make sure that the tweaking of parameter does not violate the narrative provided, as the spatial arrangement and scene playout must follow the narrative.
    Original DSL dict:{config}
    Narrative:{narrative}
    Metrics for the DSL dict: Minimum Euclidean gap:{metrics.get('min_euclid_dist_m')}, Time at minimum euclidean gap: {metrics.get('min_time_dist')}
    Use these metrics to adjust scenario parameters, for example, changing speeds, longitudinal or lateral gap at initialization.
    Minimum euclidean gap must be less than 2 meter for it to be considered a collision.
    Follow the time of minimum gap and determine if it is consistent with the narrative. For example, it could be that the narrative suggests collision much sooner than what is being done in the actual scene here.
    Also, avoid tweaking the quantities in a way that it leads to naive minimum gap at the beginning or end of trajectory. For example, avoid having collisions at timestep 0 or 1 or very early or timestep 359 or 412, which is at the very beginning or end of trajectory.
    Return a revised DSL in JSON format that only performs fine-tuning to the original DSL. Only change the original DSL substantially if it is violating the narrative, otherwise stick to numerical fine-tuning.
    """
    return parse_json_response(call_llm(prompt, model_name=model_name, max_tokens=5000))


def narratives_from_frame(near) -> List[str]:
    if "Narrative" not in near.columns:
        return []
    return [str(x) for x in near["Narrative"].tolist()]
