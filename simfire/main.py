from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

sim.run("5m")

sim.rendering = True
sim.run("1m")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
sim.save_spread_graph()
# Saved to the location specified in the config: simulation.sf_home

# Turn off rendering so the display disappears and the simulation continues to run in the background
sim.rendering = False