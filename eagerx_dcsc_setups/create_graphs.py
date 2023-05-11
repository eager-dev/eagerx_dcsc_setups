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
    rate = cfg["train"]["rate"]
    sensor_rate = rate
    actuator_rate = cfg["train"]["actuator_rate"]

    for setting in settings:
        graph = eagerx.Graph.create()

        # Make pendulum
        from eagerx_dcsc_setups.pendulum.objects import Pendulum

        # Select sensors, actuators and states of Pendulum

        states = ["model_state", "mass", "length", "dt", "max_speed"]
        sensors = ["x", "action_applied"]

        # Make pendulum
        pendulum = Pendulum.make(
            "pendulum", actuator_rate=actuator_rate, sensor_rate=sensor_rate, actuators=["u"], sensors=sensors, states=states,
        )
        # Set dt according to rate
        pendulum.states.dt.space.low = 1 / actuator_rate
        pendulum.states.dt.space.high = 1 / actuator_rate

        if "domain_randomization" in cfg["settings"][setting] and cfg["settings"][setting]["domain_randomization"]:
            pendulum.config.states.append("model_parameters")
            pendulum.states.mass.space.low = pendulum.states.mass.space.low * 0.95
            pendulum.states.mass.space.high = pendulum.states.mass.space.high * 1.05
            pendulum.states.length.space.low = pendulum.states.length.space.low * 0.95
            pendulum.states.length.space.high = pendulum.states.length.space.high * 1.05
        graph.add(pendulum)

        # Connect the pendulum to an action and observations
        graph.connect(action="voltage", target=pendulum.actuators.u)
        graph.connect(source=pendulum.sensors.x, observation="angle_data", window=2)
        graph.connect(source=pendulum.sensors.action_applied, observation="action_applied", skip=True)
        graph_dir = root / "exps" / "train" / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / f"graph_{setting}.yaml"
        graph.save(str(graph_path))
