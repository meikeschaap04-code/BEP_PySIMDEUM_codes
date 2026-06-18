import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pysimdeum
import xarray as xr
import datetime
import random
import gc

NUM_DAYS     = 365
NUM_PATTERNS = 25
HOUSE_TYPE   = 'two_person'

FLOW_CAPS_LPM = list(np.arange(1, 18.5, 0.5))

def _random_date_in_year(year: int = 2024) -> datetime.date:
    start = datetime.date(year, 1, 1)
    return start + datetime.timedelta(days=random.randint(0, 364))

def simulate_one_day(num_patterns: int) -> xr.DataArray:
    orig_simulate = pysimdeum.core.house.House.simulate
    def patched_simulate(self_obj, date=None, duration=None, num_patterns=100):
        if date is None:
            date = datetime.date.today()
        if duration is None:
            duration = datetime.timedelta(days=1)
        return orig_simulate(self_obj, date=date, duration=duration, num_patterns=num_patterns)
    pysimdeum.core.house.House.simulate = patched_simulate
    try:
        house = pysimdeum.built_house(house_type=HOUSE_TYPE)
        consumption = house.consumption.sel(flowtypes='totalflow')
    finally:
        pysimdeum.core.house.House.simulate = orig_simulate
    return consumption

def apply_centrale_begrenzing(consumption: xr.DataArray, cap_lps: float) -> xr.DataArray:
    def _is_volume_driven(name: str) -> bool:
        return any(x in name.lower() for x in ['wc', 'toilet', 'washingmachine', 'dishwasher', 'vaatwasser'])

    modified = consumption.copy()
    for p in consumption.coords['patterns'].values:
        pat        = consumption.sel(patterns=p)
        total_flow = pat.sum(['enduse', 'user'])
        safe_flow  = total_flow.where(total_flow > 0, other=1.0)
        scale      = xr.where(total_flow > cap_lps, cap_lps / safe_flow, 1.0)
        for enduse in consumption.coords['enduse'].values:
            if not _is_volume_driven(str(enduse)):
                for user in consumption.coords['user'].values:
                    modified.loc[dict(patterns=p, enduse=enduse, user=user)] = (
                        pat.sel(enduse=enduse, user=user) * scale
                    )
    return modified

def simulate_cap(cap_lpm: float, num_days: int, num_patterns: int) -> dict:
    cap_lps        = cap_lpm / 60.0
    daily_volumes  = []
    cumulative_avg = []

    for _ in range(num_days):
        try:
            raw = simulate_one_day(num_patterns)
            mod = apply_centrale_begrenzing(raw, cap_lps=cap_lps)
            day_L = float(
                mod.sum(['enduse', 'user'])
                   .mean('patterns')
                   .sum('time')
                   .values
            )
            daily_volumes.append(day_L)
            cumulative_avg.append(np.mean(daily_volumes))
            del raw, mod
            gc.collect()
        except Exception as e:
            print(f"    Fout bij cap={cap_lpm} L/min: {str(e)[:80]}")
            cumulative_avg.append(cumulative_avg[-1] if cumulative_avg else np.nan)

    return {
        'yearly_m3':        (np.nanmean(daily_volumes) / 1000) * 365,
        'cumulative_avg_L': cumulative_avg,
    }


print("=" * 70)
print(f"Centrale Begrenzing Sweep  |  {NUM_DAYS} dag(en), {NUM_PATTERNS} patronen/dag")
print("=" * 70)

results = {}
for cap in FLOW_CAPS_LPM:
    print(f"\n  Debiet: {cap:.1f} L/min ...", end=" ", flush=True)
    results[cap] = simulate_cap(cap, NUM_DAYS, NUM_PATTERNS)
    print(f"-> {results[cap]['yearly_m3']:.1f} m3/jaar")

caps      = list(results.keys())
yearly_m3 = [results[c]['yearly_m3'] for c in caps]

def _interp_m3(cap_target):
    return float(np.interp(cap_target, caps, yearly_m3))

TARGET_M3  = 100 * 2 * 365 / 1000
cap_100lpd = caps[int(np.argmin(np.abs(np.array(yearly_m3) - TARGET_M3)))]

MARKERS_BASE = [
    (15.6, 'red',         f'15,6 L/min: {_interp_m3(15.6):.1f} m3/jaar'),
    (17.8, 'forestgreen', f'17,8 L/min: {_interp_m3(17.8):.1f} m3/jaar'),
]
MARKER_TARGET = (cap_100lpd, 'darkorange', f'100 L/p/dag: {cap_100lpd:.1f} L/min')

def _make_sweep_plot(ax, include_target: bool):
    ax.plot(caps, yearly_m3, marker='o', linewidth=2, color='steelblue', markersize=6)
    ax.fill_between(caps, yearly_m3, alpha=0.15, color='steelblue')
    legend_handles = []
    markers = MARKERS_BASE + ([MARKER_TARGET] if include_target else [])
    for cap_mark, color, label in markers:
        y_mark = _interp_m3(cap_mark)
        handle = ax.scatter([cap_mark], [y_mark], color=color, s=80, zorder=5, label=label)
        legend_handles.append(handle)
    if include_target:
        ax.axhline(TARGET_M3, color='darkorange', linestyle=':', linewidth=1.4, alpha=0.7)
    ax.set_xlabel('Centrale begrenzing (L/min)', fontsize=12)
    ax.set_ylabel('Jaarlijks waterverbruik (m3/jaar)', fontsize=12)
    ax.set_title('Waterverbruik vs centrale begrenzing', fontsize=13)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=9,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

fig, ax = plt.subplots(figsize=(11, 5))
_make_sweep_plot(ax, include_target=False)
plt.tight_layout()
plot1_path = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\centrale_begrenzing_sweep.png'
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"\nPlot A opgeslagen: {plot1_path}")
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
_make_sweep_plot(ax, include_target=True)
plt.tight_layout()
plot2_path = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\centrale_begrenzing_sweep2.png'
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"Plot B opgeslagen: {plot2_path}")
plt.show()

if NUM_DAYS > 1:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(FLOW_CAPS_LPM)))
    for i, cap in enumerate(FLOW_CAPS_LPM):
        cum  = [(v / 1000) * 365 for v in results[cap]['cumulative_avg_L']]
        days = list(range(1, len(cum) + 1))
        ax2.plot(days, cum, linewidth=1.2, color=cmap[i], label=f'{cap} L/min')
    ax2.set_xlabel('Aantal gesimuleerde dagen', fontsize=12)
    ax2.set_ylabel('Cumulatief gemiddeld verbruik (m3/jaar)', fontsize=12)
    ax2.set_title('Convergentie per debietpunt', fontsize=13)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(fontsize=7, ncol=3, loc='upper right', title='Cap (L/min)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\centrale_begrenzing_convergentie.png'
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"Convergentieplot opgeslagen: {plot3_path}")
    plt.show()
