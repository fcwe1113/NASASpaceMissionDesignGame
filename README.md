# Space Mission Design Game

This is a hackathon project developed during the [NASA Space Apps Challenge](https://www.spaceappschallenge.org/), developed by both me and [Rohit Godhwani](https://github.com/RohitGODHWANI).

## Features

- Similar to Kerbal Space Program, you can send missions that will launch from earth to other planets
- These missions can accumulate science points to unlock better base modules and ship parts to further help your space exploration targets
- Maybe even terraform mars along the way? (to be implemented)

## How to install

1. Install uv (a Python project manager) — see the uv installation guide.
2. Download this repository (green Code button → Download ZIP, then unzip) or clone it with git.
3. In a terminal, go into the game folder and run: `uv run main.py`

The first run downloads Python (if needed) and the game’s libraries; after that it starts straight away.

Optional: playing beyond the year 2150. The game ships with de440s.bsp (planet positions for 1849–2150). For dates up to 2650, download [de440.bsp](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp) (about 115 MB) into the game folder; it’s picked up automatically.

## AI use disclaimer

Claude is used to vibe code the game, however I deliberately chose a way of presentation that does not require generating assets for the game to work

## Special Thanks

- Kerbal space program for providing the sample for the spacecraft navigation portion of the game, buy the game [here](https://store.steampowered.com/app/220200/Kerbal_Space_Program/), it is VERY good
- Solar Expanse for the logistics mechanics of the game, it recently went on early access on steam at April 2026, see their store page [here](https://store.steampowered.com/app/1369700/Solar_Expanse_Space_Exploration_Manager/)