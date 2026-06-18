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
print("Water Saving Scenarios Analysis")
print("=" * 80)

NUM_DAYS     = 500
NUM_PATTERNS = 25
HOUSE_TYPE   = 'family'

def _detect_num_persons(house_type: str) -> int:
    house = pysimdeum.built_house(house_type=house_type)
    users = list(house.consumption.coords['user'].values)
    persons = [u for u in users if 'household' not in str(u).lower()]
    n = len(persons)
    print(f"Gedetecteerd: {n} personen in huishoudtype '{house_type}' (users: {users})")
    return n

NUM_PERSONS = _detect_num_persons(HOUSE_TYPE)

SCENARIOS = {
    'Nulmeting': {
        'description': f'Standaard {HOUSE_TYPE} huishouden (geen aanpassingen)',
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
    volume_keywords = [
        'wc', 'toilet',
        'washingmachine', 'washing_machine',
        'dishwasher', 'vaatwasser',
        'bath', 'bad',
        'cook', 'kook',
        'garden', 'tuin', 'outdoor', 'buiten',
        'car', 'auto',
        'pool', 'zwembad',
        'boiler', 'cv',
    ]
    return any(x in name for x in volume_keywords)

def apply_centrale_begrenzing(consumption: xr.DataArray, cap_lps: float) -> xr.DataArray:
    modified = consumption.copy()
    enduses      = consumption.coords['enduse'].values
    vol_enduses  = [e for e in enduses if     _is_volume_driven(e)]
    tijd_enduses = [e for e in enduses if not _is_volume_driven(e)]
    for p in consumption.coords['patterns'].values:
        pat = consumption.sel(patterns=p)
        vol_flow  = (pat.sel(enduse=vol_enduses).sum(['enduse', 'user'])
                     if vol_enduses else xr.zeros_like(pat.isel(enduse=0, user=0)))
        tijd_flow = (pat.sel(enduse=tijd_enduses).sum(['enduse', 'user'])
                     if tijd_enduses else xr.zeros_like(pat.isel(enduse=0, user=0)))
        available  = (cap_lps - vol_flow).clip(min=0.0)
        tijd_scale = xr.where(tijd_flow > available, available / tijd_flow, 1.0)
        vol_scale  = xr.where(vol_flow > cap_lps, cap_lps / vol_flow, 1.0)
        for enduse in enduses:
            scale = vol_scale if _is_volume_driven(enduse) else tijd_scale
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

def _random_date_in_year(year: int = 2024) -> datetime.date:
    start = datetime.date(year, 1, 1)
    delta = (datetime.date(year, 12, 31) - start).days
    return start + datetime.timedelta(days=random.randint(0, delta))

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
        house       = pysimdeum.built_house(house_type=HOUSE_TYPE)
        consumption = house.consumption.sel(flowtypes='totalflow')
    finally:
        pysimdeum.core.house.House.simulate = orig_simulate
    return consumption

def run_scenario(name: str, cfg: Dict, num_days: int, num_patterns: int) -> Dict:
    print(f"\n  Scenario: {name}")
    daily_volumes        = []
    daily_peak_flows_lpm = []
    cumulative_avg_L     = []
    cap_diagnostics      = []
    daily_enduse_volumes: Dict[str, list] = {}

    for day_idx in range(num_days):
        date = _random_date_in_year()
        try:
            raw = simulate_one_day(date, num_patterns)
            if (cfg['modifications'] is not None
                    and cfg['modifications']['type'] == 'centrale_begrenzing'):
                cap_lps   = cfg['modifications']['cap_lps']
                raw_total = raw.sum(['enduse', 'user'])
                frac      = float((raw_total > cap_lps).mean())
                cap_diagnostics.append(frac)

            mod        = apply_modifications(raw, cfg['modifications'])
            total_flow = mod.sum(['enduse', 'user']).mean('patterns')

            daily_volumes.append(float(total_flow.sum('time').values))
            cumulative_avg_L.append(np.mean(daily_volumes))
            daily_peak_flows_lpm.append(float(total_flow.max('time').values) * 60.0)

            enduse_flow = mod.sum('user').mean('patterns')
            enduse_vol  = enduse_flow.sum('time')
            for eu in enduse_vol.coords['enduse'].values:
                key = str(eu)
                daily_enduse_volumes.setdefault(key, []).append(
                    float(enduse_vol.sel(enduse=eu).values)
                )

            del raw, mod
            gc.collect()
        except Exception as e:
            print(f"    Fout dag {day_idx + 1}: {str(e)[:80]}")
            cumulative_avg_L.append(cumulative_avg_L[-1] if cumulative_avg_L else np.nan)
            daily_peak_flows_lpm.append(daily_peak_flows_lpm[-1] if daily_peak_flows_lpm else np.nan)

        if (day_idx + 1) % max(1, num_days // 10) == 0 or day_idx == 0:
            avg_peak = np.nanmean(daily_peak_flows_lpm)
            max_peak = np.nanmax(daily_peak_flows_lpm)
            cap_info = (f"  |  cap actief: {np.mean(cap_diagnostics)*100:.1f}%"
                        if cap_diagnostics else "")
            print(
                f"    Dag {day_idx + 1}/{num_days}  |  "
                f"gem. verbruik: {cumulative_avg_L[-1]:.1f} L/dag  |  "
                f"gem. piek: {avg_peak:.2f} L/min  |  "
                f"max piek: {max_peak:.2f} L/min{cap_info}"
            )

    mean_daily_L       = np.nanmean(daily_volumes)
    yearly_m3          = (mean_daily_L / 1000) * 365
    mean_peak_flow_lpm = np.nanmean(daily_peak_flows_lpm)
    max_peak_flow_lpm  = np.nanmax(daily_peak_flows_lpm)

    if cap_diagnostics:
        avg_cap_frac = np.mean(cap_diagnostics) * 100
        print(f"\n    Cap gemiddeld actief in {avg_cap_frac:.2f}% van alle tijdstappen")
        if avg_cap_frac < 2.0:
            print(f"    Let op: cap zelden overschreden — besparing zal klein zijn.")

    return {
        'yearly_m3':            yearly_m3,
        'mean_daily_L':         mean_daily_L,
        'mean_peak_flow_lpm':   mean_peak_flow_lpm,
        'max_peak_flow_lpm':    max_peak_flow_lpm,
        'daily_volumes':        daily_volumes,
        'daily_peak_flows_lpm': daily_peak_flows_lpm,
        'cumulative_avg_L':     cumulative_avg_L,
        'cap_active_pct':       np.mean(cap_diagnostics) * 100 if cap_diagnostics else None,
        'mean_enduse_daily_L':  {eu: np.nanmean(vols) for eu, vols in daily_enduse_volumes.items()},
    }


if __name__ == "__main__":
    print(f"\nInstellingen: {NUM_DAYS} dag(en), {NUM_PATTERNS} patronen/dag, "
          f"huis: {HOUSE_TYPE}, personen: {NUM_PERSONS}\n")

    results = {}
    for name, cfg in SCENARIOS.items():
        results[name] = run_scenario(name, cfg, NUM_DAYS, NUM_PATTERNS)

    print("\n" + "=" * 108)
    print("RESULTATEN")
    print("=" * 108)

    baseline_m3 = results['Nulmeting']['yearly_m3']
    print(
        f"\n{'Scenario':<20} {'m³/jaar':>9}  {'L/p/dag':>8}  "
        f"{'Gem. piek (L/min)':>18}  {'Max. piek (L/min)':>18}  "
        f"{'Cap actief':>11}  {'Besparing':>12}"
    )
    print("-" * 108)

    for name, res in results.items():
        saving_m3   = baseline_m3 - res['yearly_m3']
        saving_pct  = saving_m3 / baseline_m3 * 100 if baseline_m3 else 0
        tag         = "BASELINE" if name == 'Nulmeting' else f"{saving_m3:+.2f} m³ ({saving_pct:+.1f}%)"
        l_per_p_day = res['mean_daily_L'] / NUM_PERSONS
        cap_str     = (f"{res['cap_active_pct']:.1f}%"
                       if res.get('cap_active_pct') is not None else "-")
        print(
            f"{name:<20} {res['yearly_m3']:>9.2f}  "
            f"{l_per_p_day:>8.1f}  "
            f"{res['mean_peak_flow_lpm']:>18.2f}  "
            f"{res['max_peak_flow_lpm']:>18.2f}  "
            f"{cap_str:>11}  "
            f"{tag:>12}"
        )

    all_enduses    = sorted({eu for res in results.values() for eu in res['mean_enduse_daily_L']})
    scenario_names = list(results.keys())
    col_w          = 13
    sep_width      = 28 + col_w * len(scenario_names)

    print("\n" + "=" * sep_width)
    print("WATERVERBRUIK PER APPARAAT  (L/dag gemiddeld, gehele simulatieperiode)")
    print("=" * sep_width)
    print(f"{'Apparaat':<28}" + "".join(f"{n:>{col_w}}" for n in scenario_names))
    print("-" * sep_width)
    for eu in all_enduses:
        row = f"{eu:<28}"
        for name in scenario_names:
            val = results[name]['mean_enduse_daily_L'].get(eu, 0.0)
            row += f"{val:>{col_w}.2f}"
        print(row)
    print("-" * sep_width)
    totals = f"{'TOTAAL':<28}"
    for name in scenario_names:
        totals += f"{results[name]['mean_daily_L']:>{col_w}.2f}"
    print(totals)
    print(f"\n(Eenheid: liter per dag per huishouden — {NUM_PERSONS} personen)\n")

    def _style_ax(ax):
        ax.set_xlabel('# gesimuleerde dagen', fontsize=12)
        ax.set_ylabel('Cumulatief gemiddeld waterverbruik (m³/jaar)', fontsize=12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        days = list(range(1, len(res['cumulative_avg_L']) + 1))
        cum_avg_m3_yr = [(v / 1000) * 365 for v in res['cumulative_avg_L']]
        ax.plot(days, cum_avg_m3_yr, marker='o', markersize=2, linewidth=1.5, label=name)
    ax.set_title(f'Convergentie waterverbruik per scenario: {HOUSE_TYPE} eerste 100 dagen', fontsize=13)
    ax.set_xlim(1, 100)
    _style_ax(ax)
    plt.tight_layout()
    plot_path_1 = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\convergentie_100dagen7juni.png'
    plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
    print(f"\nPlot 1 opgeslagen: {plot_path_1}")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        days = list(range(1, len(res['cumulative_avg_L']) + 1))
        cum_avg_m3_yr = [(v / 1000) * 365 for v in res['cumulative_avg_L']]
        ax.plot(days, cum_avg_m3_yr, marker='o', markersize=1, linewidth=1.5, label=name)
    ax.set_title(f'Convergentie waterverbruik per scenario: {HOUSE_TYPE}', fontsize=13)
    _style_ax(ax)
    plt.tight_layout()
    plot_path_2 = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\convergentie_alledagen7juni.png'
    plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
    print(f"Plot 2 opgeslagen: {plot_path_2}")
    plt.show()
