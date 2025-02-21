import random
import yaml
import os

def generate_yaml(i):
    data = {
        "area": {
        "screen_size": [450, 450],
        "pixel_scale": 50,
        },
        "display": {
            "fire_size": 2,
            "control_line_size": 2,
            "agent_size": 4,
            "rescale_factor": 2,
        },
        "simulation": {
            "update_rate": 1,
            "runtime": f"{random.randint(10, 30)}h",  # Random runtime between 10 and 30 hours
            "headless": False,
            "draw_spread_graph": False,
            "record": True,
            "save_data": True,
            "data_type": "npy",
            "sf_home": "~/.simfire",
        },
        "mitigation": {
            "ros_attenuation": random.choice([True, False]),  # Random attenuation based on fire conditions
        },
        "operational": {
            "latitude": random.uniform(30, 50),  # Latitude between 30 and 50
            "longitude": random.uniform(-125, -105),  # Longitude between -125 and -105
            "height": 2000,
            "width": 2000,
            "resolution": 30,
            "year": random.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024]),
        },
        "terrain": {
            "topography": {
                "type": "operational",
                "functional": {
                    "function": random.choice(["perlin", "gaussian"]),  # Random choice for topography function
                    "perlin": {
                        "octaves": random.randint(3, 6),  # Random octaves between 3 and 6
                        "persistence": round(random.uniform(0.5, 0.8), 2),  # Random persistence between 0.5 and 0.8
                        "lacunarity": round(random.uniform(1.5, 3.0), 2),  # Random lacunarity between 1.5 and 3.0
                        "seed": random.randint(0, 10000),
                        "range_min": random.uniform(50.0, 150.0),  # Random range_min between 50.0 and 150.0
                        "range_max": random.uniform(150.0, 400.0),  # Random range_max between 150.0 and 400.0
                    },
                    "gaussian": {
                        "amplitude": random.randint(200, 600),  # Random amplitude between 200 and 600
                        "mu_x": random.randint(20, 80),  # Random mu_x between 20 and 80
                        "mu_y": random.randint(20, 80),  # Random mu_y between 20 and 80
                        "sigma_x": random.randint(40, 100),  # Random sigma_x between 40 and 100
                        "sigma_y": random.randint(40, 100),  # Random sigma_y between 40 and 100
                    },
                },
            },
            "fuel": {
                "type": "operational",
                "functional": {
                    "function": "chaparral",
                    "chaparral": {
                        "seed": random.randint(0, 10000),
                    },
                },
                "burn_probability": {
                    "type": "operational",
                    "functional": {
                        "function": random.choice(["perlin", "gaussian"]),  # Random choice for burn probability function
                        "perlin": {
                            "octaves": random.randint(3, 6),
                            "persistence": round(random.uniform(0.5, 0.8), 2),
                            "lacunarity": round(random.uniform(1.5, 3.0), 2),
                            "seed": random.randint(0, 5000),
                            "range_min": random.randint(50, 150),
                            "range_max": random.randint(200, 400),
                        },
                        "gaussian": {
                            "amplitude": random.randint(200, 600),
                            "mu_x": random.randint(20, 80),
                            "mu_y": random.randint(20, 80),
                            "sigma_x": random.randint(40, 100),
                            "sigma_y": random.randint(40, 100),
                        },
                    },
                },
            },
        },
        "fire": {
            "fire_initial_position": {
                "type": "static",
                "static": {
                    "position": [random.randint(5, 45), random.randint(5, 45)],  # Random fire position within the grid
                },
                "random": {
                    "seed": random.randint(1000, 10000),  # Random seed for random positioning
                },
            },
            "max_fire_duration": random.randint(1, 120),  # Random fire duration between 1 and 120 minutes
            "diagonal_spread": random.choice([True, False]),  # Random diagonal spread decision
        },
        "environment": {
            "moisture": round(random.uniform(0.001, 0.01), 3),  # Random moisture between 0.001 and 0.01
        },
        "wind": {
            "function": random.choice(["perlin", "simple"]),  # Random wind function (perlin or simple)
            "cfd": {
                "time_to_train": random.randint(500, 1500),  # Random training time
                "result_accuracy": 1,
                "iterations": random.randint(1, 5),  # Random iterations between 1 and 5
                "scale": random.randint(1, 3),  # Random scale factor
                "timestep_dt": random.uniform(0.5, 2.0),  # Random time step dt
                "diffusion": random.uniform(0.0, 1.0),  # Random diffusion value
                "viscosity": random.uniform(0.0000001, 0.0001),  # Random viscosity value
                "speed": random.randint(5, 30),  # Random wind speed between 5 and 30
                "direction": random.choice(["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]),
            },
            "simple": {
                "speed": random.randint(0, 90),  # Random wind speed for simple wind model
                "direction": random.randint(0, 360),  # Random wind direction
            },
            "perlin": {
                "speed": {
                    "seed": random.randint(0, 10000),
                    "scale": random.randint(100, 500),  # Random scale for perlin noise
                    "octaves": random.randint(3, 6),
                    "persistence": round(random.uniform(0.5, 0.8), 2),
                    "lacunarity": round(random.uniform(1.5, 3.0), 2),
                    "range_min": random.randint(7, 15),
                    "range_max": random.randint(30, 50),
                },
                "direction": {
                    "seed": random.randint(1000, 5000),
                    "scale": random.randint(1000, 2000),  # Random scale for perlin direction
                    "octaves": random.randint(2, 5),
                    "persistence": round(random.uniform(0.8, 1.0), 2),
                    "lacunarity": round(random.uniform(1.0, 2.0), 2),
                    "range_min": random.uniform(0.0, 90.0),  # Random range for wind direction
                    "range_max": random.uniform(270.0, 360.0),
                },
            },
        },
    }
    os.makedirs('configs', exist_ok=True)
    filename = f"configs/operational_config{i}.yaml"
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    print(f"Generated {filename}")
for i in range(50):
    generate_yaml(i)
