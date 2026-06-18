import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import pysimdeum
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict
import datetime
import random
import gc

print("=" * 80)
print("Water Saving Scenarios - Two Person Household Analysis")
print("=" * 80)

NUM_DAYS     = 500
NUM_PATTERNS = 25
HOUSE_TYPE   = 'two_person'

SUMMER_MONTHS = [6, 7, 8]
WINTER_MONTHS = [12, 1, 2]

SCENARIOS = {
    'Nulmeting': {
        'description': 'Standaard two-person huishouden (geen aanpassingen)',
        'modifications': None,
    },
    'Scenario 1a': {
        'description': 'Hoofdaansluiting begrensd op 15,6 L/min (EU waterkeur)',
        'modifications': {'type': 'centrale_begrenzing', 'cap_lps': 15.6 / 60.0},
    },
    'Scenario 1b': {
        'description': 'Hoofdaansluiting begrensd op 17,8 L/min (comfort)',
        'modifications': {'type': 'centrale_begrenzing', 'cap_lps': 17.8 / 60.0},
    },
    'Scenario 1c': {
        'description': 'Hoofdaansluiting begrensd op 5,0 L/min (streng)',
        'modifications': {'type': 'centrale_begrenzing', 'cap_lps': 5.0 / 60.0},
    },
    'Scenario 2': {
        'description': 'Decentrale restrictors: kranen 5 L/min, douche 7,8 L/min',
        'modifications': {
            'type': 'decentraal',
            'tap_cap_lps':    5.0 / 60.0,
            'shower_cap_lps': 7.8 / 60.0,
        },
    },
    'Scenario 3': {
        'description': 'WC 2,4L, douche max 7,4 L/min, kraan max 8 L/min',
        'modifications': {
            'type': 'waterbesparend',
            'wc_cistern_liter': 2.4,
            'shower_cap_lps':   7.4 / 60.0,
            'tap_cap_lps':      8.0 / 60.0,
        },
    },
    'Scenario 4': {
        'description': 'WC 2,4L + decentrale restrictors (kranen 5, douche 7,8 L/min)',
        'modifications': {
            'type': 'hybride',
            'wc_cistern_liter': 2.4,
            'tap_cap_lps':      5.0 / 60.0,
            'shower_cap_lps':   7.8 / 60.0,
        },
    },
}

def _is_volume_driven(enduse_name: str) -> bool:
    name = str(enduse_name).lower()
    return any(x in name for x in ['wc', 'toilet', 'washingmachine', 'dishwasher', 'vaatwasser'])

def apply_centrale_begrenzing(consumption: xr.DataArray, cap_lps: float) -> xr.DataArray:
    modified = consumption.copy()
    for p in consumption.coords['patterns'].values:
        pat = consumption.sel(patterns=p)
        total_flow = pat.sum(['enduse', 'user'])
        scale = xr.where(total_flow > cap_lps, cap_lps / total_flow, 1.0)
        for enduse in consumption.coords['enduse'].values:
            if not _is_volume_driven(enduse):
                for user in consumption.coords['user'].values:
                    modified.loc[dict(patterns=p, enduse=enduse, user=user)] = (
                        pat.sel(enduse=enduse, user=user) * scale
                    )
    return modified

def apply_decentrale_begrenzing(consumption: xr.DataArray,
                                tap_cap_lps: float,
                                shower_cap_lps: float) -> xr.DataArray:
    modified = consumption.copy()
    for enduse in consumption.coords['enduse'].values:
        name = str(enduse).lower()
        if _is_volume_driven(name):
            continue
        if 'shower' in name:
            cap = shower_cap_lps
        elif 'tap' in name:
            cap = tap_cap_lps
        else:
            continue
        modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=cap)
    return modified

def apply_waterbesparend(consumption: xr.DataArray,
                         wc_cistern_liter: float,
                         shower_cap_lps: float,
                         tap_cap_lps: float) -> xr.DataArray:
    wc_volume_factor = wc_cistern_liter / 6.0
    modified = consumption.copy()
    for enduse in consumption.coords['enduse'].values:
        name = str(enduse).lower()
        if 'wc' in name or 'toilet' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse) * wc_volume_factor
        elif 'shower' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=shower_cap_lps)
        elif 'tap' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=tap_cap_lps)
    return modified

def apply_hybride(consumption: xr.DataArray,
                  wc_cistern_liter: float,
                  tap_cap_lps: float,
                  shower_cap_lps: float) -> xr.DataArray:
    wc_volume_factor = wc_cistern_liter / 6.0
    modified = consumption.copy()
    for enduse in consumption.coords['enduse'].values:
        name = str(enduse).lower()
        if 'wc' in name or 'toilet' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse) * wc_volume_factor
        elif 'shower' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=shower_cap_lps)
        elif 'tap' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=tap_cap_lps)
    return modified

def apply_modifications(consumption: xr.DataArray, modifications: Dict) -> xr.DataArray:
    if modifications is None:
        return consumption
    mod_type = modifications['type']
    if mod_type == 'centrale_begrenzing':
        return apply_centrale_begrenzing(consumption, cap_lps=modifications['cap_lps'])
    elif mod_type == 'decentraal':
        return apply_decentrale_begrenzing(
            consumption,
            tap_cap_lps=modifications['tap_cap_lps'],
            shower_cap_lps=modifications['shower_cap_lps'],
        )
    elif mod_type == 'waterbesparend':
        return apply_waterbesparend(
            consumption,
            wc_cistern_liter=modifications['wc_cistern_liter'],
            shower_cap_lps=modifications['shower_cap_lps'],
            tap_cap_lps=modifications['tap_cap_lps'],
        )
    elif mod_type == 'hybride':
        return apply_hybride(
            consumption,
            wc_cistern_liter=modifications['wc_cistern_liter'],
            tap_cap_lps=modifications['tap_cap_lps'],
            shower_cap_lps=modifications['shower_cap_lps'],
        )
    return consumption

def _last_day_of_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31

def _random_date_in_season(season: str, year: int = 2024) -> datetime.date:
    if season == 'summer':
        months = SUMMER_MONTHS
    elif season == 'winter':
        months = WINTER_MONTHS
    elif season == 'year':
        months = list(range(1, 13))
    else:
        raise ValueError(f"Onbekend seizoen: '{season}'. Kies 'summer', 'winter' of 'year'.")
    month = random.choice(months)
    day = random.randint(1, _last_day_of_month(year, month))
    return datetime.date(year, month, day)

def simulate_one_day(date: datetime.date, num_patterns: int) -> xr.DataArray:
    orig_simulate = pysimdeum.core.house.House.simulate
    target = num_patterns
    def patched_simulate(self_obj, date=None, duration=None, num_patterns=100):
        if date is None:
            date = datetime.date.today()
        if duration is None:
            duration = datetime.timedelta(days=1)
        return orig_simulate(self_obj, date=date, duration=duration, num_patterns=target)
    pysimdeum.core.house.House.simulate = patched_simulate
    try:
        house = pysimdeum.built_house(house_type=HOUSE_TYPE)
        consumption = house.consumption.sel(flowtypes='totalflow')
    finally:
        pysimdeum.core.house.House.simulate = orig_simulate
    return consumption

def run_scenario(name: str, cfg: Dict, num_days: int, num_patterns: int,
                 season: str) -> Dict:
    print(f"\n  [{season.upper()}] Scenario: {name}")
    daily_volumes        = []
    daily_peak_flows_lpm = []
    cumulative_avg_L     = []

    for day_idx in range(num_days):
        date = _random_date_in_season(season)
        try:
            raw = simulate_one_day(date, num_patterns)
            mod = apply_modifications(raw, cfg['modifications'])
            total_flow = mod.sum(['enduse', 'user']).mean('patterns')

            daily_volumes.append(float(total_flow.sum('time').values))
            cumulative_avg_L.append(np.mean(daily_volumes))
            daily_peak_flows_lpm.append(float(total_flow.max('time').values) * 60.0)

            del raw, mod
            gc.collect()
        except Exception as e:
            print(f"    Fout dag {day_idx + 1}: {str(e)[:80]}")
            cumulative_avg_L.append(cumulative_avg_L[-1] if cumulative_avg_L else np.nan)
            daily_peak_flows_lpm.append(daily_peak_flows_lpm[-1] if daily_peak_flows_lpm else np.nan)

        if (day_idx + 1) % max(1, num_days // 10) == 0 or day_idx == 0:
            avg_peak = np.nanmean(daily_peak_flows_lpm)
            max_peak = np.nanmax(daily_peak_flows_lpm)
            print(
                f"    Dag {day_idx + 1}/{num_days}  |  "
                f"gem. verbruik: {cumulative_avg_L[-1]:.1f} L/dag  |  "
                f"gem. piek: {avg_peak:.2f} L/min  |  "
                f"max piek: {max_peak:.2f} L/min"
            )

    mean_daily_L = np.nanmean(daily_volumes)
    return {
        'yearly_m3':            (mean_daily_L / 1000) * 365,
        'mean_daily_L':         mean_daily_L,
        'mean_peak_flow_lpm':   np.nanmean(daily_peak_flows_lpm),
        'max_peak_flow_lpm':    np.nanmax(daily_peak_flows_lpm),
        'daily_volumes':        daily_volumes,
        'daily_peak_flows_lpm': daily_peak_flows_lpm,
        'cumulative_avg_L':     cumulative_avg_L,
    }

def _style_ax(ax, season_label: str):
    ax.set_xlabel('# gesimuleerde dagen', fontsize=12)
    ax.set_ylabel('Cumulatief gemiddeld waterverbruik (m³/jaar)', fontsize=12)
    ax.set_title(f'Convergentie waterverbruik per scenario — {season_label}', fontsize=13)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

def make_convergence_plot(results_season: Dict, season_label: str, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results_season.items():
        days = list(range(1, len(res['cumulative_avg_L']) + 1))
        cum_avg_m3_yr = [(v / 1000) * 365 for v in res['cumulative_avg_L']]
        ax.plot(days, cum_avg_m3_yr, marker='o', markersize=1, linewidth=1.5, label=name)
    _style_ax(ax, season_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot opgeslagen: {save_path}")
    plt.show()

def print_results_table(results: Dict, season_label: str):
    baseline_m3 = results['Nulmeting']['yearly_m3']
    NUM_PERSONS  = 2
    print(f"\n{'=' * 90}")
    print(f"RESULTATEN — {season_label}")
    print("=" * 90)
    print(
        f"\n{'Scenario':<20} {'m³/jaar':>9}  {'L/p/dag':>8}  "
        f"{'Gem. piek (L/min)':>18}  {'Max. piek (L/min)':>18}  {'Besparing':>12}"
    )
    print("-" * 94)
    for name, res in results.items():
        saving_m3  = baseline_m3 - res['yearly_m3']
        saving_pct = saving_m3 / baseline_m3 * 100 if baseline_m3 else 0
        tag        = "BASELINE" if saving_pct == 0 else f"{saving_m3:+.2f} m³ ({saving_pct:+.1f}%)"
        print(
            f"{name:<20} {res['yearly_m3']:>9.2f}  "
            f"{res['mean_daily_L'] / NUM_PERSONS:>8.1f}  "
            f"{res['mean_peak_flow_lpm']:>18.2f}  "
            f"{res['max_peak_flow_lpm']:>18.2f}  "
            f"{tag:>12}"
        )


if __name__ == "__main__":
    print(f"\nInstellingen: {NUM_DAYS} dag(en) per periode, {NUM_PATTERNS} patronen/dag, huis: {HOUSE_TYPE}")
    print("\nSeizoendefinities (meteorologisch, KNMI):")
    print("  Zomer  : juni, juli, augustus")
    print("  Winter : december, januari, februari")
    print("  Jaar   : alle maanden\n")

    seasons = {
        'summer': {
            'label':     'Zomer (jun–aug)',
            'plot_path': r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\convergentie_zomer.png',
        },
        'winter': {
            'label':     'Winter (dec–feb)',
            'plot_path': r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\convergentie_winter.png',
        },
        'year': {
            'label':     'Heel jaar (jan–dec)',
            'plot_path': r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\convergentie_jaargem.png',
        },
    }

    all_results = {}
    for season_key, season_info in seasons.items():
        print(f"\n{'#' * 60}")
        print(f"  SEIZOEN: {season_info['label']}")
        print(f"{'#' * 60}")
        season_results = {}
        for name, cfg in SCENARIOS.items():
            season_results[name] = run_scenario(
                name, cfg, NUM_DAYS, NUM_PATTERNS, season=season_key
            )
        all_results[season_key] = season_results
        print_results_table(season_results, season_info['label'])
        make_convergence_plot(season_results, season_info['label'], season_info['plot_path'])

    print("\nKlaar. Plots opgeslagen:")
    for s, info in seasons.items():
        print(f"  {info['label']}: {info['plot_path']}")
