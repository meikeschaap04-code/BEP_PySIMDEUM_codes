import warnings, time, datetime, random, gc
import numpy as np
import xarray as xr
import pysimdeum
import pysimdeum.core.house
import matplotlib.pyplot as plt
warnings.simplefilter(action="ignore", category=FutureWarning)

NUM_DAYS        = 300
NUM_PATTERNS    = 25
HOUSE_TYPE      = 'two_person'
SEWER_THRESH_LS = 0.10
PLOT_PATH = r'C:\Users\meike\OneDrive\Documenten\Claude\Projects\Bep Meike\afvalwater_pieken_500d.png'

# Drainagefracties per eindgebruik
DRAINAGE = {
    'shower':         1.00,
    'bath':           1.00,
    'bathtub':        1.00,
    'bathroomtap':    1.00,
    'wc':             1.00,
    'toilet':         1.00,
    'dishwasher':     1.00,
    'washingmachine': 1.00,
    'kitchentap':     0.85,
    'outsidetap':     0.00,
    'outsidertap':    0.00,
}

def drainage_frac(enduse: str) -> float:
    name = str(enduse).lower().replace('_', '').replace(' ', '')
    for key, f in DRAINAGE.items():
        if key in name:
            return f
    return 1.0

SCENARIOS = {
    'Nulmeting':   {'modifications': None},
    'Scenario 1a': {'modifications': {'type': 'centrale_begrenzing',
                                      'cap_lps': 15.6 / 60}},
    'Scenario 1b': {'modifications': {'type': 'centrale_begrenzing',
                                      'cap_lps': 17.8 / 60}},
    'Scenario 2':  {'modifications': {'type': 'decentraal',
                                      'tap_cap_lps': 5.0 / 60,
                                      'shower_cap_lps': 7.8 / 60}},
    'Scenario 3':  {'modifications': {'type': 'waterbesparend',
                                      'wc_cistern_liter': 2.4,
                                      'shower_cap_lps': 7.4 / 60,
                                      'tap_cap_lps': 8.0 / 60}},
    'Scenario 4':  {'modifications': {'type': 'hybride',
                                      'wc_cistern_liter': 2.4,
                                      'tap_cap_lps': 5.0 / 60,
                                      'shower_cap_lps': 7.8 / 60}},
    'Scenario 5a': {'modifications': {'type': 'centrale_decentraal',
                                      'cap_lps': 15.6 / 60,
                                      'tap_cap_lps': 5.0 / 60,
                                      'shower_cap_lps': 7.8 / 60}},
    'Scenario 5b': {'modifications': {'type': 'centrale_decentraal',
                                      'cap_lps': 17.8 / 60,
                                      'tap_cap_lps': 5.0 / 60,
                                      'shower_cap_lps': 7.8 / 60}},
}

SCENARIO_COLORS = {
    'Nulmeting':   '#4C72B0',
    'Scenario 1a': '#DD8452',
    'Scenario 1b': '#C44E52',
    'Scenario 2':  '#55A868',
    'Scenario 3':  '#8172B3',
    'Scenario 4':  '#937860',
    'Scenario 5a': '#DA8BC3',
    'Scenario 5b': '#8C8C8C',
}

def _is_volume_driven(enduse_name: str) -> bool:
    name = str(enduse_name).lower()
    return any(x in name for x in [
        'wc', 'toilet', 'washingmachine', 'washing_machine',
        'dishwasher', 'vaatwasser', 'bath', 'bad',
        'cook', 'garden', 'tuin', 'outdoor', 'buiten',
    ])

def apply_centrale_begrenzing(consumption, cap_lps):
    modified = consumption.copy()
    enduses  = consumption.coords['enduse'].values
    vol_eu   = [e for e in enduses if     _is_volume_driven(e)]
    tijd_eu  = [e for e in enduses if not _is_volume_driven(e)]
    for p in consumption.coords['patterns'].values:
        pat = consumption.sel(patterns=p)
        vol_flow  = (pat.sel(enduse=vol_eu).sum(['enduse', 'user'])
                     if vol_eu else xr.zeros_like(pat.isel(enduse=0, user=0)))
        tijd_flow = (pat.sel(enduse=tijd_eu).sum(['enduse', 'user'])
                     if tijd_eu else xr.zeros_like(pat.isel(enduse=0, user=0)))
        available  = (cap_lps - vol_flow).clip(min=0.0)
        tijd_scale = xr.where(tijd_flow > available, available / tijd_flow, 1.0)
        vol_scale  = xr.where(vol_flow  > cap_lps,   cap_lps  / vol_flow,  1.0)
        for enduse in enduses:
            scale = vol_scale if _is_volume_driven(enduse) else tijd_scale
            for user in consumption.coords['user'].values:
                modified.loc[dict(patterns=p, enduse=enduse, user=user)] = (
                    pat.sel(enduse=enduse, user=user) * scale
                )
    return modified

def apply_decentrale_begrenzing(consumption, tap_cap_lps, shower_cap_lps):
    modified = consumption.copy()
    for enduse in consumption.coords['enduse'].values:
        name = str(enduse).lower()
        if _is_volume_driven(name):
            continue
        if   'shower' in name: cap = shower_cap_lps
        elif 'tap'    in name: cap = tap_cap_lps
        else: continue
        modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=cap)
    return modified

def apply_waterbesparend(consumption, wc_cistern_liter, shower_cap_lps, tap_cap_lps):
    wc_factor = wc_cistern_liter / 6.0
    modified  = consumption.copy()
    for enduse in consumption.coords['enduse'].values:
        name = str(enduse).lower()
        if   'wc' in name or 'toilet' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse) * wc_factor
        elif 'shower' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=shower_cap_lps)
        elif 'tap' in name:
            modified.loc[dict(enduse=enduse)] = consumption.sel(enduse=enduse).clip(max=tap_cap_lps)
    return modified

def apply_hybride(consumption, wc_cistern_liter, tap_cap_lps, shower_cap_lps):
    return apply_waterbesparend(consumption, wc_cistern_liter, shower_cap_lps, tap_cap_lps)

def apply_centrale_decentraal(consumption, cap_lps, tap_cap_lps, shower_cap_lps):
    return apply_decentrale_begrenzing(
        apply_centrale_begrenzing(consumption, cap_lps),
        tap_cap_lps, shower_cap_lps
    )

def apply_modifications(consumption, modifications):
    if modifications is None:
        return consumption
    t = modifications['type']
    if   t == 'centrale_begrenzing': return apply_centrale_begrenzing(consumption, modifications['cap_lps'])
    elif t == 'decentraal':          return apply_decentrale_begrenzing(consumption, modifications['tap_cap_lps'], modifications['shower_cap_lps'])
    elif t == 'waterbesparend':      return apply_waterbesparend(consumption, modifications['wc_cistern_liter'], modifications['shower_cap_lps'], modifications['tap_cap_lps'])
    elif t == 'hybride':             return apply_hybride(consumption, modifications['wc_cistern_liter'], modifications['tap_cap_lps'], modifications['shower_cap_lps'])
    elif t == 'centrale_decentraal': return apply_centrale_decentraal(consumption, modifications['cap_lps'], modifications['tap_cap_lps'], modifications['shower_cap_lps'])
    return consumption

def simulate_one_day(date, num_patterns):
    orig   = pysimdeum.core.house.House.simulate
    target = num_patterns
    def patched(self, date=None, duration=None, num_patterns=100):
        return orig(self,
                    date=date or datetime.date.today(),
                    duration=duration or datetime.timedelta(days=1),
                    num_patterns=target)
    pysimdeum.core.house.House.simulate = patched
    try:
        house = pysimdeum.built_house(house_type=HOUSE_TYPE)
        return house.consumption.sel(flowtypes='totalflow')
    finally:
        pysimdeum.core.house.House.simulate = orig

def _random_date():
    return datetime.date(2024, 1, 1) + datetime.timedelta(days=random.randint(0, 364))

def compute_sewer_flow(mod: xr.DataArray) -> xr.DataArray:
    enduses = mod.coords['enduse'].values
    fracs   = xr.DataArray(
        [drainage_frac(e) for e in enduses],
        dims=['enduse'], coords={'enduse': enduses}
    )
    return (mod * fracs).sum(['enduse', 'user']).mean('patterns')


if __name__ == '__main__':
    t0 = time.perf_counter()
    print("=" * 70)
    print("Afvalwater Piekanalyse")
    print("=" * 70)
    print(f"Instellingen: {NUM_DAYS} dagen | {NUM_PATTERNS} patronen | {HOUSE_TYPE}\n")

    results = {}
    for name, cfg in SCENARIOS.items():
        print(f"\n{name}")
        daily_peaks     = []
        subthresh_fracs = []
        daily_profiles  = []
        all_flows       = []

        for day_idx in range(NUM_DAYS):
            try:
                raw   = simulate_one_day(_random_date(), NUM_PATTERNS)
                mod   = apply_modifications(raw, cfg['modifications'])
                sewer = compute_sewer_flow(mod)
                vals  = sewer.values

                daily_peaks.append(float(vals.max()))
                daily_profiles.append(vals.copy())

                active   = vals > 0
                frac_sub = (vals[active] < SEWER_THRESH_LS).mean() if active.any() else 0.0
                subthresh_fracs.append(frac_sub)

                active_vals = vals[vals > 0]
                if len(active_vals) > 0:
                    all_flows.append(active_vals)

                del raw, mod, sewer
                gc.collect()
            except Exception as e:
                print(f"  Fout dag {day_idx + 1}: {e}")

            if (day_idx + 1) % 50 == 0:
                print(f"  dag {day_idx + 1:3d}/{NUM_DAYS} | "
                      f"gem. piek: {np.mean(daily_peaks):.3f} L/s | "
                      f"P95 piek:  {np.percentile(daily_peaks, 95):.3f} L/s")

        results[name] = {
            'mean_peak_ls':   np.mean(daily_peaks),
            'p95_peak_ls':    np.percentile(daily_peaks, 95),
            'max_peak_ls':    np.max(daily_peaks),
            'mean_subthresh': np.mean(subthresh_fracs) * 100,
            'daily_peaks':    daily_peaks,
            'mean_profile':   np.mean(daily_profiles, axis=0),
            'all_flows':      np.concatenate(all_flows),
        }

    print("\n" + "=" * 85)
    print("RESULTATEN")
    print("=" * 85)

    baseline_mean = results['Nulmeting']['mean_peak_ls']
    baseline_p95  = results['Nulmeting']['p95_peak_ls']

    print(f"\n{'Scenario':<14} {'Gem. piek':>10} {'P95 piek':>10} {'Max piek':>10} "
          f"{'Red. gem.':>10} {'Red. P95':>10} {'< drempel':>11}")
    print(f"{'':14} {'(L/s)':>10} {'(L/s)':>10} {'(L/s)':>10} "
          f"{'':>10} {'':>10} {'act. tijd':>11}")
    print("-" * 85)

    for name, res in results.items():
        red_mean = (baseline_mean - res['mean_peak_ls']) / baseline_mean * 100
        red_p95  = (baseline_p95  - res['p95_peak_ls'])  / baseline_p95  * 100
        tag_m    = "BASELINE" if name == 'Nulmeting' else f"{red_mean:+.1f}%"
        tag_p    = "BASELINE" if name == 'Nulmeting' else f"{red_p95:+.1f}%"
        print(f"{name:<14} {res['mean_peak_ls']:>10.3f} {res['p95_peak_ls']:>10.3f} "
              f"{res['max_peak_ls']:>10.3f} {tag_m:>10} {tag_p:>10} "
              f"{res['mean_subthresh']:>10.1f}%")

    elapsed = time.perf_counter() - t0
    print(f"\nRuntime: {elapsed / 60:.1f} min")

    # Plots
    scenario_names     = list(results.keys())
    scenario_names_no5 = [n for n in scenario_names if n not in ('Scenario 5a', 'Scenario 5b')]
    colors             = [SCENARIO_COLORS[n] for n in scenario_names]
    colors_no5         = [SCENARIO_COLORS[n] for n in scenario_names_no5]

    def save_and_show(fig, path):
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot opgeslagen: {path}")
        plt.show()

    def plot_boxbar(names, cols, path, title_suffix=''):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle(f'Afvalwater piekdebiet per scenario{title_suffix}',
                     fontsize=13, y=1.01)

        ax = axes[0]
        bp = ax.boxplot(
            [results[n]['daily_peaks'] for n in names],
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
            widths=0.55,
        )
        for patch, color in zip(bp['boxes'], cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for flier, color in zip(bp['fliers'], cols):
            flier.set(markerfacecolor=color, markeredgecolor=color)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Piekdebiet afvoer (L/s)', fontsize=10)
        ax.set_title('Verdeling dagelijkse piekdebieten', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax2  = axes[1]
        bp95 = results['Nulmeting']['p95_peak_ls']
        reds = [(bp95 - results[n]['p95_peak_ls']) / bp95 * 100 for n in names]
        bars = ax2.barh(names[::-1], reds[::-1], color=cols[::-1],
                        alpha=0.85, height=0.55, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, reds[::-1]):
            label = "BASELINE" if abs(val) < 0.05 else f"{val:.1f}%"
            if val >= 0:
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         label, va='center', ha='left', fontsize=9, color='black')
            elif val > -20:
                ax2.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
                         label, va='center', ha='right', fontsize=9, color='black')
            else:
                ax2.text(-2, bar.get_y() + bar.get_height() / 2,
                         label, va='center', ha='right', fontsize=9, color='white')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Reductie piekdebiet t.o.v. nulmeting (%)', fontsize=10)
        ax2.set_title('Piekdebietsreductie', fontsize=11)
        ax2.grid(axis='x', alpha=0.3, linewidth=0.8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(left=min(reds) - 5, right=max(reds) + 8)
        save_and_show(fig, path)

    plot_boxbar(scenario_names, colors, PLOT_PATH)
    plot_boxbar(scenario_names_no5, colors_no5,
                PLOT_PATH.replace('.png', '_zonder5.png'),
                title_suffix=' (zonder 5a/5b)')

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for name in scenario_names_no5:
        profile = results[name]['mean_profile']
        t_hours = np.linspace(0, 24, len(profile))
        ax3.plot(t_hours, profile,
                 label=name, color=SCENARIO_COLORS[name], linewidth=1.5)
    ax3.set_xlabel('Tijd van de dag (uur)', fontsize=10)
    ax3.set_ylabel('Gem. afvoerdebiet (L/s)', fontsize=10)
    ax3.set_title('Gemiddeld dagprofiel afvoerdebiet per scenario', fontsize=12)
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 2))
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(alpha=0.3, linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    save_and_show(fig3, PLOT_PATH.replace('.png', '_dagprofiel.png'))

    n_scenarios = len(scenario_names_no5)
    n_cols      = 2
    n_rows      = (n_scenarios + 1) // n_cols
    y_max       = max(results[n]['mean_profile'].max() for n in scenario_names_no5) * 1.1
    fig4, axes4 = plt.subplots(n_rows, n_cols,
                                figsize=(13, n_rows * 3.2),
                                sharex=True, sharey=True)
    fig4.suptitle('Gemiddeld dagprofiel afvoerdebiet per scenario', fontsize=13, y=1.01)
    axes4_flat = axes4.flatten()
    for i, name in enumerate(scenario_names_no5):
        ax  = axes4_flat[i]
        profile = results[name]['mean_profile']
        t_hours = np.linspace(0, 24, len(profile))
        ax.plot(t_hours, profile, color=SCENARIO_COLORS[name], linewidth=1.5)
        ax.fill_between(t_hours, profile, alpha=0.15, color=SCENARIO_COLORS[name])
        ax.set_title(name, fontsize=10, fontweight='bold', color=SCENARIO_COLORS[name])
        ax.set_xlim(0, 24)
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i % n_cols == 0:
            ax.set_ylabel('Afvoerdebiet (L/s)', fontsize=9)
        if i >= n_scenarios - n_cols:
            ax.set_xlabel('Tijd van de dag (uur)', fontsize=9)
    for j in range(n_scenarios, len(axes4_flat)):
        axes4_flat[j].set_visible(False)
    save_and_show(fig4, PLOT_PATH.replace('.png', '_dagprofiel_apart.png'))

    fig5, ax5 = plt.subplots(figsize=(14, 6))
    bp5 = ax5.boxplot(
        [results[n]['all_flows'] for n in scenario_names],
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=2, alpha=0.15, linestyle='none'),
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(bp5['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for flier, color in zip(bp5['fliers'], colors):
        flier.set(markerfacecolor=color, markeredgecolor=color)
    ax5.set_xticks(range(1, len(scenario_names) + 1))
    ax5.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=9)
    ax5.set_ylabel('Afvoerdebiet (L/s)', fontsize=10)
    ax5.set_title('Volledige debietdistributie per scenario', fontsize=11)
    ax5.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    save_and_show(fig5, PLOT_PATH.replace('.png', '_volledige_range.png'))

    fig6, ax6 = plt.subplots(figsize=(12, 6))
    bp6 = ax6.boxplot(
        [results[n]['all_flows'] for n in scenario_names_no5],
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=2, alpha=0.15, linestyle='none'),
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(bp6['boxes'], colors_no5):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for flier, color in zip(bp6['fliers'], colors_no5):
        flier.set(markerfacecolor=color, markeredgecolor=color)
    ax6.set_xticks(range(1, len(scenario_names_no5) + 1))
    ax6.set_xticklabels(scenario_names_no5, rotation=30, ha='right', fontsize=9)
    ax6.set_ylabel('Afvoerdebiet (L/s)', fontsize=10)
    ax6.set_title('Volledige debietdistributie per scenario', fontsize=11)
    ax6.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    save_and_show(fig6, PLOT_PATH.replace('.png', '_volledige_range_zonder5.png'))
