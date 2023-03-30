import eagerx
from pathlib import Path
import yaml


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path("/home/jelle/eagerx_dev/eagerx_dcsc_setups")

    # Load config
    cfg_path = root / "cfg" / "train.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    settings = cfg["settings"]

    for setting in settings:
        rate = cfg["settings"][setting]["rate"]
        sensor_rate = rate
        actuator_rate = cfg["settings"][setting]["actuator_rate"]

        graph = eagerx.Graph.create()

        # Make pendulum
        from eagerx_dcsc_setups.pendulum.objects import Pendulum

        # Select sensors, actuators and states of Pendulum
        states = ["model_state", "mass", "length", "max_speed"]
        sensors = ["x"]
        if "action_applied" in cfg["settings"][setting]:
            if cfg["settings"][setting]["action_applied"]:
                sensors.append("action_applied")
        if "window_x" in cfg["settings"][setting]:
            window_x = cfg["settings"][setting]["window_x"]
        else:
            window_x = 1


        # Make pendulum
        pendulum = Pendulum.make(
            "pendulum", actuator_rate=actuator_rate, sensor_rate=sensor_rate, actuators=["u"], sensors=sensors, states=states
        )
        graph.add(pendulum)

        # Connect the pendulum to an action and observations
        graph.connect(action="voltage", target=pendulum.actuators.u)
        graph.connect(source=pendulum.sensors.x, observation="angle_data", window=window_x)
        if "action_applied" in sensors:
            graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", skip=True)
        graph_dir = root / "exps" / "train" / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / f"graph_{setting}.yaml"
        graph.save(str(graph_path))
