"""
Cross-Feature Explorer Tab Renderer

Interactive tab for exploring relationships between any pair of features,
flavor co-occurrence, and multi-way interaction analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from page_apps.analytics.common import (
    CRAMERS_NOTE, EFFECT_METRIC_LABELS, ETA2_NOTE, collapsed_text, effect_metric_of, fmt_n, fmt_p, fmt_q,
)


def render_cross_feature_tab():
    """Render the Cross-Feature Explorer tab"""
    st.header("Cross-Feature Explorer")

    st.markdown("""
    **What you'll see here:** Relationships between features, flavor co-occurrence,
    and multi-way interaction effects (how feature combinations produce unexpected
    flavor profiles or price premiums).
    """)

    from analytics.frontend.cached_data_loader import (
        load_cross_feature_data, load_cooccurrence_data, load_interaction_data
    )
    cross_data = load_cross_feature_data()
    cooccurrence_data = load_cooccurrence_data()
    interaction_data = load_interaction_data()

    section = st.radio(
        "Mode:",
        [
            "Discovery (associations)",
            "Flavor Co-occurrence",
            "Flavor Interactions",
            "Price Interactions",
        ],
        horizontal=True,
        key="xf_section",
    )

    if section == "Discovery (associations)":
        _render_discovery(cross_data)
    elif section == "Flavor Co-occurrence":
        _render_cooccurrence(cooccurrence_data)
    elif section == "Flavor Interactions":
        _render_flavor_interactions(interaction_data)
    elif section == "Price Interactions":
        _render_price_interactions(interaction_data)


# --------------------------------------------------------------------------
# Discovery mode
# --------------------------------------------------------------------------

def _association_notes(row: Dict[str, Any]) -> str:
    """Short per-row note: chi-square status, or flavors tested for mean-V rows."""
    notes = []
    status = row.get('test_status')
    if status == 'monte_carlo':
        notes.append("Monte Carlo p (2,000 simulations)")
    elif status == 'collapsed':
        notes.append("small categories combined")
    coll = collapsed_text(row.get('collapsed'), None)
    if coll and status in ('collapsed', 'monte_carlo'):
        notes.append("categories merged into 'Other'")
    if row.get('n_flavors_tested') is not None:
        notes.append(f"{row['n_flavors_tested']} of {row.get('n_flavors_total', row['n_flavors_tested'])} flavors tested")
    return "; ".join(notes)


def _render_discovery(data: Dict[str, Any]):
    """Show all feature-pair associations, one chart per effect-size metric"""
    if not data:
        st.info("Cross-feature data not available. Please regenerate the analytics cache.")
        return

    associations = data.get('all_associations', [])
    if not associations:
        st.info("No feature-pair associations computed yet.")
        return

    st.subheader("Feature-Pair Association Strength")
    st.write("Each kind of test has its own effect-size scale, so they are charted separately "
             "(effect sizes from different tests are not comparable with each other):")

    assoc_df = pd.DataFrame(associations)
    assoc_df['metric'] = [effect_metric_of(r) for r in associations]
    assoc_df['pair'] = assoc_df.apply(lambda r: f"{r['feature_a']} x {r['feature_b']}", axis=1)

    order = ['cramers_v', 'eta2_h', 'mean_cramers_v']
    metrics = [m for m in order if m in set(assoc_df['metric'])] + \
              [m for m in assoc_df['metric'].unique() if m not in order]
    for metric in metrics:
        sub = assoc_df[assoc_df['metric'] == metric].sort_values('effect_size', ascending=False)
        label = EFFECT_METRIC_LABELS.get(metric, metric)
        fig = px.bar(
            sub, x='pair', y='effect_size',
            title=f"Association strength: {label}",
            labels={'pair': 'Feature Pair', 'effect_size': label},
            text=sub['effect_size'].apply(lambda x: f"{x:.3f}"),
        )
        fig.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({
        'Pair': assoc_df['pair'],
        'Test': assoc_df['test'],
        'Effect Size': assoc_df['effect_size'],
        'Metric': assoc_df['metric'].map(lambda m: EFFECT_METRIC_LABELS.get(m, m)),
        'p': [fmt_p(r.get('p_value'), r.get('test_status'),
                    2000 if r.get('test_status') == 'monte_carlo' else None) for r in associations],
        'q-value': [fmt_q(r.get('q_value')) for r in associations],
        'Significant?': [("Yes" if r.get('is_significant') else "No") if r.get('is_significant') is not None
                         else "n/a" for r in associations],
        'N': [fmt_n(r.get('n_observations', '')) if r.get('n_observations') is not None else '' for r in associations],
        'Notes': [_association_notes(r) for r in associations],
    })
    st.dataframe(table, hide_index=True)

    insufficient = data.get('insufficient_association_tests') or []
    if insufficient:
        st.info("Not enough data to test: " + "; ".join(
            f"{t.get('feature_a')} x {t.get('feature_b')}" for t in insufficient) + ".")

    st.info(
        "**Why different tests?** The system automatically picks the right test for each data type:  \n"
        "- **Chi-square test** → used when both features are categories (e.g., country × process). "
        "It checks whether the combination of categories occurs more or less often than random chance would predict. "
        "Where the table has many small counts, small categories are combined (or a Monte Carlo p-value is used) so the test stays valid. "
        + CRAMERS_NOTE + "  \n"
        "- **Kruskal-Wallis H test** → used when one feature is a category and the other is a number (e.g., country × price). "
        "It's a non-parametric alternative to ANOVA — it asks \"do these groups have different distributions?\" without assuming the data is normally distributed. "
        + ETA2_NOTE + "  \n"
        "- **Mean Cramér's V** → used when one feature is a list (e.g., flavors) and the other is a category. "
        "Each individual flavor is tested for association with the category, then the effect sizes are averaged across the flavors tested "
        "(no single p-value).  \n\n"
        "Effect size tells you *how strong* the relationship is. The **q-value** is the p-value adjusted for the number of "
        "pairs tested; q < 0.05 is treated as significant."
    )


# --------------------------------------------------------------------------
# Flavor Co-occurrence
# --------------------------------------------------------------------------

def _render_cooccurrence(data: Dict[str, Any]):
    """Render flavor co-occurrence analysis"""
    if not data:
        st.info("Co-occurrence data not available. Please regenerate the analytics cache.")
        return

    level = st.selectbox(
        "Taxonomy level:",
        ["Family", "Genus", "Species"],
        key="xf_cooc_level",
    )

    level_key = f'{level.lower()}_level'
    level_data = data.get(level_key, {})

    if not level_data:
        st.info(f"No co-occurrence data at {level.lower()} level.")
        return

    st.caption(
        "**PMI (Pointwise Mutual Information)** measures how much more often two flavors "
        "appear together than you'd expect by chance. PMI > 0 means they pair up more than expected; "
        "PMI > 1 means a strong, notable pairing. Negative PMI means the flavors tend to avoid each other."
    )

    distinctive = level_data.get('distinctive_combinations', {})
    if isinstance(distinctive, dict):
        surprising = distinctive.get('surprising_pairs', [])
        avoiding = distinctive.get('avoiding_pairs', [])

        if surprising:
            st.subheader("Most Surprising Pairings")
            st.write("Flavor pairs that appear together **more** than expected:")
            surp_df = pd.DataFrame(surprising[:15])
            if not surp_df.empty:
                fig = px.bar(
                    surp_df,
                    x=surp_df.apply(
                        lambda r: f"{r['flavor_1']} + {r['flavor_2']}", axis=1
                    ),
                    y='pmi',
                    title=f"Top Surprising Flavor Pairs ({level} Level)",
                    labels={'x': 'Flavor Pair', 'pmi': 'PMI Score'},
                    text=surp_df['cooccurrence_count'].apply(lambda x: f'n={fmt_n(x)}'),
                    color='pmi',
                    color_continuous_scale='Greens',
                )
                fig.update_layout(xaxis_tickangle=45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rarely Co-occurring Pairs")
        if avoiding and 'expected_count' in avoiding[0]:
            st.write("Flavor pairs that appear together **less** than chance would predict "
                     "(including pairs that never appear together):")
            for pair in avoiding[:8]:
                f1 = pair.get('flavor_1', '')
                f2 = pair.get('flavor_2', '')
                obs = pair.get('cooccurrence_count', 0)
                exp = pair.get('expected_count', 0)
                st.write(
                    f"- **{f1}** + **{f2}**: together in {fmt_n(obs)} coffees, "
                    f"vs about {exp:.0f} expected (q = {pair.get('q_value', 0):.1e})"
                )
            st.caption("All pairs of flavors listed in at least 3 coffees each were tested "
                       "(one-sided Fisher exact test, corrected for multiple comparisons).")
        elif avoiding:
            st.info("Rarely-co-occurring pairs are not in this version of the cache yet. "
                    "They will appear after the cache is regenerated.")
        else:
            st.info("No pairs of flavors appear together significantly less often than expected.")

    top_pairs = level_data.get('top_pairs_by_count', [])
    if top_pairs:
        st.subheader("Most Common Flavor Pairs")
        pairs_df = pd.DataFrame(top_pairs[:20])
        if not pairs_df.empty:
            pairs_df['pair'] = pairs_df.apply(
                lambda r: f"{r['flavor_1']} + {r['flavor_2']}", axis=1
            )
            fig = px.bar(
                pairs_df,
                x='pair',
                y='cooccurrence_count',
                title=f"Most Frequent Flavor Pairs ({level} Level)",
                labels={'pair': 'Flavor Pair', 'cooccurrence_count': 'Times Seen Together'},
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    matrix_data = level_data.get('matrix', {})
    if matrix_data and matrix_data.get('labels') and matrix_data.get('values'):
        st.subheader("Co-occurrence Matrix")
        labels = matrix_data['labels']
        values = matrix_data['values']
        if len(labels) <= 25:
            fig = px.imshow(
                values, x=labels, y=labels,
                title=f"Flavor Co-occurrence ({level} Level)",
                color_continuous_scale='YlOrRd', aspect='auto',
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"Matrix has {len(labels)} flavors — too large for a heatmap. "
                "Use the Family level for a compact view."
            )


# --------------------------------------------------------------------------
# Flavor Interactions
# --------------------------------------------------------------------------

def _render_flavor_interactions(data: Dict[str, Any]):
    """Render multi-way flavor interaction analysis"""
    if not data:
        st.info("Interaction data not available. Please regenerate the analytics cache.")
        return

    st.subheader("Flavor Interactions")
    st.markdown("""
    Discover **emergent** flavors that appear more often in a specific combination
    than either feature alone would predict, and **suppressed** flavors that
    unexpectedly disappear.
    """)
    st.caption(
        "**How it works:** If Ethiopian coffees have Fruity 60% of the time, and Washed coffees "
        "have Fruity 40% of the time, we'd *expect* Ethiopian Washed coffees to have Fruity at a "
        "rate based on those individual rates. If the actual rate is much higher, Fruity is "
        "\"emergent\" for that combination — the pairing creates something unexpected. "
        "The **Interaction Score** = observed rate - expected rate."
    )
    st.caption(
        "**How we test it:** for each combination and flavor we ask whether the observed number of coffees "
        "with the flavor is more (or less) than the expected rate would produce (a binomial test). "
        "Every combination-and-flavor tested at once is counted when adjusting for multiple comparisons "
        "(Benjamini-Hochberg q-value; q < 0.05 is significant). "
        "Combinations involving a varietal use single-varietal coffees only (coffees listing several varietals are left out)."
    )

    interaction_type = st.selectbox(
        "Interaction type:",
        [
            "Origin x Process -> Flavor",
            "Process x Varietal -> Flavor",
            "Origin x Varietal -> Flavor",
            "Three-Way (Origin x Process x Varietal)",
        ],
        key="fi_type",
    )

    type_map = {
        "Origin x Process -> Flavor": ('origin_process_flavor', False),
        "Process x Varietal -> Flavor": ('process_varietal_flavor', False),
        "Origin x Varietal -> Flavor": ('origin_varietal_flavor', False),
        "Three-Way (Origin x Process x Varietal)": ('three_way_flavor', True),
    }

    prefix, is_3way = type_map[interaction_type]

    if is_3way:
        level = "Family"
        cache_key = 'three_way_flavor_family'
    else:
        level = st.selectbox(
            "Taxonomy level:", ["Family", "Genus"], key="fi_level"
        )
        cache_key = f'{prefix}_{level.lower()}'

    idata = data.get(cache_key, {})
    if not idata or not idata.get('has_data'):
        st.info("Not enough data for this interaction analysis.")
        return

    st.caption(
        f"**{idata.get('valid_combinations', 0)}** valid combinations found "
        f"(min {10} coffees each)"
    )

    # Top emergent flavors
    emergent = idata.get('top_emergent', [])
    if emergent:
        st.subheader("Top Emergent Flavors")
        st.write("Flavors that appear **more** than expected in these combos:")
        _render_interaction_effects_chart(emergent[:15], 'emergent', is_3way)

    # Top suppressed flavors
    suppressed = idata.get('top_suppressed', [])
    if suppressed:
        st.subheader("Top Suppressed Flavors")
        st.write("Flavors that appear **less** than expected:")
        _render_interaction_effects_chart(suppressed[:15], 'suppressed', is_3way)

    # Drill-down by combination
    profiles = idata.get('combination_profiles', {})
    if profiles:
        st.subheader("Drill Down by Combination")
        combo_keys = sorted(profiles.keys(), key=lambda k: -profiles[k]['sample_size'])
        combo_labels = [
            f"{k.replace('|', ' + ')} (n={fmt_n(profiles[k]['sample_size'])})"
            for k in combo_keys
        ]
        selected_idx = st.selectbox(
            "Select combination:", range(len(combo_labels)),
            format_func=lambda i: combo_labels[i], key="fi_combo"
        )
        if selected_idx is not None:
            combo_key = combo_keys[selected_idx]
            _render_combo_flavor_profile(profiles[combo_key], idata)


def _render_interaction_effects_chart(effects: List[Dict], effect_type: str, is_3way: bool):
    """Render bar chart of interaction effects"""
    df = pd.DataFrame(effects)
    if df.empty:
        return

    if is_3way:
        df['combo'] = df.apply(
            lambda r: f"{r['country']} + {r['process']} + {r['varietal']}", axis=1
        )
    else:
        df['combo'] = df.apply(
            lambda r: f"{r['feature_a_value']} + {r['feature_b_value']}", axis=1
        )

    df['label'] = df.apply(
        lambda r: f"{r['combo']}: {r['flavor']} (n={fmt_n(r['sample_size'])})", axis=1
    )

    color_scale = 'Greens' if effect_type == 'emergent' else 'Reds'
    fig = px.bar(
        df,
        x='interaction_score',
        y='label',
        orientation='h',
        title=f"{'Emergent' if effect_type == 'emergent' else 'Suppressed'} Flavor Effects",
        labels={'interaction_score': 'Interaction Score (observed - expected)', 'label': ''},
        color='interaction_score',
        color_continuous_scale=color_scale,
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=max(400, len(df) * 30))
    st.plotly_chart(fig, use_container_width=True)

    # Table
    with st.expander("Detailed data"):
        table_cols = ['combo', 'flavor', 'observed_rate', 'expected_rate',
                      'interaction_score', 'sample_size', 'q_value', 'is_significant']
        available = [c for c in table_cols if c in df.columns]
        st.dataframe(
            df[available].rename(columns={
                'combo': 'Combination',
                'flavor': 'Flavor',
                'observed_rate': 'Observed Rate',
                'expected_rate': 'Expected Rate',
                'interaction_score': 'Interaction',
                'sample_size': 'N',
                'q_value': 'q-value',
                'is_significant': 'Significant?',
            }),
            hide_index=True,
        )


def _render_combo_flavor_profile(profile: Dict, idata: Dict):
    """Render the full flavor profile for a selected combination"""
    st.write(f"**Sample size:** {profile['sample_size']} coffees")
    if profile['sample_size'] < 20:
        st.warning("Small sample size — interpret with caution.")

    flavors = profile.get('flavors', [])
    if not flavors:
        return

    fdf = pd.DataFrame(flavors)

    fig = px.bar(
        fdf,
        x='flavor',
        y='observed_rate',
        title="Flavor Profile",
        labels={'observed_rate': 'Proportion', 'flavor': 'Flavor'},
        text=fdf['count'].apply(lambda x: f'n={fmt_n(x)}'),
    )
    fig.update_layout(xaxis_tickangle=45, yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Price Interactions
# --------------------------------------------------------------------------

def _render_price_interactions(data: Dict[str, Any]):
    """Render multi-way price interaction analysis"""
    if not data:
        st.info("Interaction data not available. Please regenerate the analytics cache.")
        return

    st.subheader("Price Interactions")
    st.markdown("""
    Discover **premium synergies** where a combination of features commands a
    higher price than expected from each feature alone, and **discounts** where
    the combination is cheaper than expected.
    """)
    st.caption(
        "**How it works:** If Colombian coffees cost a median of $7/lb, and Washed coffees cost $6.50/lb, "
        "we calculate an expected price for Colombian Washed based on these individual prices and the "
        "global median. If the actual median is $9/lb, the +$1.50 difference is the \"interaction premium\" — "
        "the combination commands a price premium beyond what either feature alone explains."
    )
    st.caption(
        "**How we test it:** a permutation test of the premium against this additive expectation: "
        "we shuffle one feature among coffees many times (2,000 permutations) to see how often a premium this "
        "large would appear by chance. The p-values are adjusted for the number of combinations tested "
        "(Benjamini-Hochberg q-value; q < 0.05 is significant). "
        "Combinations involving a varietal use single-varietal coffees only."
    )

    price_type = st.selectbox(
        "Interaction type:",
        [
            "Origin + Flavor -> Price",
            "Origin + Process -> Price",
            "Origin + Varietal -> Price",
            "Flavor + Process -> Price",
            "Flavor + Varietal -> Price",
        ],
        key="pi_type",
    )

    key_map = {
        "Origin + Flavor -> Price": 'origin_flavor_price',
        "Origin + Process -> Price": 'origin_process_price',
        "Origin + Varietal -> Price": 'origin_varietal_price',
        "Flavor + Process -> Price": 'flavor_process_price',
        "Flavor + Varietal -> Price": 'flavor_varietal_price',
    }

    cache_key = key_map[price_type]
    pdata = data.get(cache_key, {})

    if not pdata or not pdata.get('has_data'):
        st.info("Not enough data for this price interaction analysis.")
        return

    global_med = pdata.get('global_median_price', 0)
    st.caption(
        f"**{pdata.get('valid_combinations', 0)}** valid combinations | "
        f"Global median: ${global_med:.2f}/lb (price per lb = best per-lb price offered, usually the largest bag)"
    )

    # Top premiums
    premiums = pdata.get('top_premiums', [])
    if premiums:
        st.subheader("Biggest Price Premiums")
        st.write("Combinations priced **higher** than expected:")
        _render_price_effects_chart(premiums[:20], 'premium')

    # Top discounts
    discounts = pdata.get('top_discounts', [])
    if discounts:
        st.subheader("Biggest Price Discounts")
        st.write("Combinations priced **lower** than expected:")
        _render_price_effects_chart(discounts[:20], 'discount')

    # Full data table
    all_combos = pdata.get('combinations', [])
    if all_combos:
        with st.expander(f"All {len(all_combos)} combinations"):
            cdf = pd.DataFrame(all_combos)
            display_cols = {
                'feature_a_value': pdata.get('feature_a_label', 'Feature A'),
                'feature_b_value': pdata.get('feature_b_label', 'Feature B'),
                'sample_size': 'N',
                'median_price': 'Median $/lb',
                'expected_price': 'Expected $/lb',
                'price_premium': 'Premium',
                'q_value': 'q-value',
                'is_significant': 'Significant?',
            }
            available = {k: v for k, v in display_cols.items() if k in cdf.columns}
            st.dataframe(
                cdf[list(available.keys())].rename(columns=available),
                hide_index=True,
            )


def _render_price_effects_chart(effects: List[Dict], effect_type: str):
    """Render bar chart of price premiums/discounts"""
    df = pd.DataFrame(effects)
    if df.empty:
        return

    df['combo'] = df.apply(
        lambda r: f"{r['feature_a_value']} + {r['feature_b_value']}", axis=1
    )
    df['label'] = df.apply(
        lambda r: f"{r['combo']} (n={fmt_n(r['sample_size'])}, ${r['median_price']:.2f}/lb"
                  + (f", q={fmt_q(r['q_value'])}" if r.get('q_value') is not None and pd.notna(r.get('q_value')) else "")
                  + ")",
        axis=1,
    )

    color_col = 'price_premium'
    color_scale = 'Greens' if effect_type == 'premium' else 'Reds_r'

    fig = px.bar(
        df,
        x='price_premium',
        y='label',
        orientation='h',
        title=f"{'Premium Synergies' if effect_type == 'premium' else 'Discount Effects'}",
        labels={'price_premium': 'Price Premium ($/lb vs expected)', 'label': ''},
        color=color_col,
        color_continuous_scale=color_scale,
        text=df['price_premium'].apply(lambda x: f"${x:+.2f}"),
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        height=max(400, len(df) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)
