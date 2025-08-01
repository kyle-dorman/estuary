# Estuary Remote Sensing

Repo for estuaries

## Set me up

Some one time repo initialization work. 


Clone repo
```bash
git clone git@github.com:kyledorman/estuary.git
```

#### Mac Dependancy Installation

Install packages through brew
```bash
brew doctor
brew update
brew upgrade
brew install cairo gdal clang uv
```

## Build Me

Repo uses uv

```bash
uv lock
```

### Update Dependencies

To update the dependencies add/delete/update the pyproject.toml file and run
```bash
uv lock
```

### Jupyter Me

To launch jupyter, run
```bash
./start_jupyter.sh
```

### Lint Me

To format code run
```bash
./lint.sh
```

### Tensorboard Me

To launch jupyter, run
```bash
./tensorboard_start.sh
```

### Run Me
```bash
uv run --env-file .env scripts/SOMETHING.py
```

## Mouth-State Label Set  
| Code | State | Definition | Image cues |
|------|-------|------------|------------|
| **0** | **Unsure / Data gap** | Berm/bridge throat can’t be evaluated. | Cloud/fog/shadow/sensor streaks or off-frame hides ≥50% of the berm/bridge throat. |
| **1** | **Closed** | Berm (or culvert/embankment at a bridge) blocks surface flow. Lagoon water and ocean are separated by ≥1 px of dry/damp light sand with no aligned wet-sand corridor. | A continuous light/tan ridge across the mouth; dark lagoon water stops landward of the crest. Seaward side shows only normal beach wetness/foam, not aligned with the notch. At bridges: no dark/wet corridor reappears on the ocean side. |
| **2** | **Open** | Berm is broken and there is a surface connection from lagoon to ocean even if the channel is diffuse. Connection can be (a) a visible dark ribbon or (b) a wet-sand corridor aligned with the notch that reaches the swash zone. | Any of the following qualifies: Unbroken dark channel lagoon→surf; dark water blending into white surf; aligned wet-sand corridor extending from the notch to the swash (often darker or glossier than surrounding wet sand); a seaward reappearance of dark water/wet corridor beyond a bridge. Berm shoulders may remain. |
---

### Bridge / culvert rule (applies to 1 vs 2)
•	If you can trace an aligned dark or wet-sand corridor through/under the bridge that reappears seaward and reaches the swash → Open (2).
•	If water disappears under the bridge and no aligned corridor is visible seaward → Closed (1).
•	If the bridge shadow/clouds hide both inlet and outlet → Unsure (0).

### Quick decision flow
1.	Obscured throat? → 0
2.	Is there a continuous sand/structure ridge with a ≥1 px light-sand gap between lagoon water and ocean and no aligned wet-sand corridor? → 1
3.	Otherwise, berm is broken and any of these makes it open → 2
•	Visible dark ribbon to surf or dark ribbon that merges into white foam
•	Aligned wet-sand corridor from the notch to the swash (no dry gap), even if the channel isn’t visibly dark in NIR
•	At bridges, corridor reappears seaward and reaches the swash

### Notes & tie-breakers for faint flows
•	Aligned wet-sand corridor: look for a tonally darker/glossier strip from the notch trending straight to the swash; often extends deeper up-beach than adjacent wet areas.
•	Exclude normal tidal wetness: broad, alongshore-parallel wet bands without a clear connection line from the notch do not count as open.
•	Specular/glint can lighten water; rely on alignment + continuity more than tone when the ribbon isn’t dark.
•	Keep the ≥1 px dry/light sand gap rule for Closed—if any contiguous wet corridor bridges that gap, it’s Open.

uv run --env-file .env scripts/gpt.py -d /Users/kyledorman/data/estuary/label_studio/00025/ --limit 50 --dim 512 -s /Users/kyledorman/data/estuary/label_studio/00025/pred512_4o.csv --model gpt-4o; uv run --env-file .env scripts/gpt.py -d /Users/kyledorman/data/estuary/label_studio/00025/ --limit 50 --dim 256 -s /Users/kyledorman/data/estuary/label_studio/00025/pred256_4o.csv --model gpt-4o; uv run --env-file .env scripts/gpt.py -d /Users/kyledorman/data/estuary/label_studio/00025/ --limit 50 --dim 512 -s /Users/kyledorman/data/estuary/label_studio/00025/pred512_4o_high.csv --model gpt-4o --detail high; uv run --env-file .env scripts/gpt.py -d /Users/kyledorman/data/estuary/label_studio/00025/ --limit 50 --dim 256 -s /Users/kyledorman/data/estuary/label_studio/00025/pred256_4o_high.csv --model gpt-4o --detail high;