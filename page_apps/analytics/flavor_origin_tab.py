"""
Flavor & Origin tab: Explore by Origin, Explore by Flavor, Compare.

Built on the per-coffee distinctiveness analysis. For a group of coffees (a country,
region or seller) a flavor is "distinctive" when it is listed far more often than in
other coffees (at least 1.5x as often, statistically reliable after correcting for
multiple comparisons).
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.constants import unit_label, unit_plural
from analytics.frontend.cached_data_loader import (
    get_available_units, get_unit_sizes, load_cooccurrence_data, load_distinctiveness_meta,
    load_flavor_hierarchies, load_flavor_unit_rows, load_unit_profile,
)
from analytics.processing.distinctiveness import jsd_between
from page_apps.analytics.common import (
    LEVELS_NOTE, REGENERATE_MSG, SELLER_SUPPORT_NOTE, UNIT_TYPE_PLURAL_LABELS,
    UNIT_TYPE_SINGULAR_LABELS, UNIT_TYPES, evidence_frame, fmt_n, fmt_pct,
    level_selector, params_caption, sentence_for, support_text,
)

MAX_SENTENCES = 10
NO_STANDOUT = "No single flavor stands out"


def render_flavor_origin_tab():
    """Consolidated tab: Explore + By Flavor + Compare in one tab with sub-sections"""
    st.header("Flavor & Origin Analysis")

    section = st.radio(
        "Analysis mode:",
        ["Explore by Origin", "Explore by Flavor", "Compare Origins"],
        horizontal=True,
        key="flavor_origin_section"
    )

    if section == "Explore by Origin":
        render_explore_section()
    elif section == "Explore by Flavor":
        render_flavor_section()
    elif section == "Compare Origins":
        render_compare_section()


# --------------------------------------------------------------------------
# Explore by Origin
# --------------------------------------------------------------------------

def top_units_by_size(unit_type: str, units: List[str], n: int = 1) -> List[str]:
    """The n units with the most coffees (ties: alphabetical); the list order of ``units`` is untouched."""
    sizes = get_unit_sizes(unit_type) or {}
    ranked = sorted(units, key=lambda u: (-sizes.get(u, 0), u))
    return ranked[:n]


def default_unit_index(unit_type: str, units: List[str]) -> int:
    """Index (in the alphabetical list) of the unit with the most coffees."""
    top = top_units_by_size(unit_type, units, 1)
    return units.index(top[0]) if top else 0


def render_explore_section():
    """Deep-dive for one country, region or seller"""
    findings = (load_distinctiveness_meta() or {}).get('key_findings') or []
    example = (f"For example: {findings[0]['finding']}." if findings and findings[0].get('finding')
               else "The claims look like \"X% of a country's coffees list a flavor vs Y% of other coffees\".")
    st.markdown(f"""
    **What you'll see here:** What makes a country, region or seller's coffees different from the rest of the market.

    **How to interpret:**
    - **Distinctive flavor**: a flavor listed in clearly more of this group's coffees than of other coffees
      (at least 1.5x as often), and statistically reliable. {example}
    - **Difference from the rest of the market**: how different the group's overall flavor mix is (0 = identical, 1 = completely different).
    - **Variety**: how evenly the group's coffees spread across flavors (0 = one flavor, 1 = evenly spread).
    - Some groups have no flavor that stands out. That is a real result, not missing data.
    """)

    col1, col2 = st.columns(2)
    with col1:
        unit_type = st.radio("Explore by:", [UNIT_TYPE_SINGULAR_LABELS[t] for t in UNIT_TYPES],
                             key="explore_unit_type").lower()
    available_units = get_available_units(unit_type)
    with col2:
        if available_units:
            selected_unit = st.selectbox(
                f"Select {unit_type}:", available_units, index=default_unit_index(unit_type, available_units),
                format_func=lambda u: unit_label(unit_type, u), key=f"explore_unit_{unit_type}")
        else:
            selected_unit = None

    level = level_selector("explore_level")

    if not available_units:
        st.info(REGENERATE_MSG)
        return

    profile = load_unit_profile(selected_unit, unit_type)
    if profile:
        display_unit_profile(profile, level)
    else:
        st.warning(f"No analysis results found for {unit_label(unit_type, selected_unit)}")


def display_unit_profile(profile: Dict[str, Any], level: str = 'genus'):
    """Display one unit's distinctive flavors at one taxonomy level"""
    label = profile.get('label') or unit_label(profile.get('unit_type', ''), profile.get('unit_name', ''))
    unit_type = profile.get('unit_type', 'country')
    st.subheader(f"Analysis for {label}")

    levels = profile.get('levels')
    if not levels or level not in levels:
        st.info(REGENERATE_MSG)
        return
    L = levels[level]

    overview = profile.get('overview', {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Coffees", fmt_n(profile.get('n_coffees', overview.get('n_coffees', 'N/A'))))
    with c2:
        st.metric(f"Distinctive {level} flavors", L.get('n_distinctive', 0),
                  help=f"Out of {L.get('n_tested', 0)} {level}-level flavors tested for this group.")
    with c3:
        jsd = L.get('jsd')
        st.metric("Difference from market", "n/a" if jsd is None else f"{jsd:.3f}",
                  help="Jensen-Shannon divergence between this group's overall flavor mix and the rest of the market (0 = identical, 1 = completely different).")
    with c4:
        div = L.get('diversity')
        st.metric("Variety", "n/a" if div is None else f"{div:.2f}",
                  help="How evenly coffees spread across flavors at this level (0 to 1).")

    related = []
    if overview.get('n_regions') is not None:
        related.append(f"{overview['n_regions']} regions")
    if overview.get('n_countries') is not None:
        related.append(f"{overview['n_countries']} countries")
    if overview.get('n_sellers') is not None:
        related.append(f"{overview['n_sellers']} sellers")
    if related:
        st.caption("Covers coffees from " + ", ".join(related) + ".")

    flavors: List[Dict[str, Any]] = L.get('flavors', [])
    if not flavors:
        st.info(f"{NO_STANDOUT} for {label} at the {level} level. "
                "Its coffees' flavors look much like the rest of the market's.")
    else:
        st.markdown(f"**Flavor signature ({level}):** " + ", ".join(L.get('signature') or [f['flavor'] for f in flavors[:3]]))
        for row in flavors[:MAX_SENTENCES]:
            line = f"- {sentence_for(row, unit_type, profile.get('unit_name'))}"
            sup = support_text(row)
            if sup:
                line += f" — {sup}"
            st.markdown(line)
        if unit_type != 'seller':
            st.caption(SELLER_SUPPORT_NOTE)
        with st.expander("Evidence for each flavor"):
            st.dataframe(evidence_frame(flavors, unit_type), hide_index=True)

    meta = load_distinctiveness_meta()
    caption = params_caption(meta)
    if caption:
        st.caption(caption.format(types=unit_plural(unit_type)) + " " + LEVELS_NOTE)


# --------------------------------------------------------------------------
# Explore by Flavor
# --------------------------------------------------------------------------

def render_flavor_section():
    """Flavor-first exploration"""
    st.markdown("""
    **What you'll see here:** Start with a specific flavor and see which countries, regions or sellers list it unusually often.

    **How to interpret:**
    - **Flavor hierarchy**: Family (broad, like "Fruity"), Genus (more specific, like "Berry"), or Species (most specific, like "Blueberry").
    - **Where it stands out**: places where far more coffees list this flavor than elsewhere, shown as "X% of this place's coffees vs Y% of other coffees".
    - **Often listed together**: other flavors that appear in the same coffees, with how often compared with all coffees that have flavor notes.

    Perfect for exploring "Where can I find the best [specific flavor]?" questions.
    """)

    flavor_data = load_flavor_hierarchies()
    if not flavor_data:
        st.warning("Flavor hierarchy data not available")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        family_options = ["All"] + sorted(flavor_data['families'])
        selected_family = st.selectbox("Flavor Family:", family_options)
    with col2:
        if selected_family != "All" and selected_family in flavor_data['genera_by_family']:
            genus_options = ["All"] + sorted(flavor_data['genera_by_family'][selected_family])
        else:
            genus_options = ["All"] + sorted(flavor_data['all_genera'])
        selected_genus = st.selectbox("Flavor Genus:", genus_options)
    with col3:
        if selected_genus != "All" and selected_genus in flavor_data['species_by_genus']:
            species_options = ["All"] + sorted(flavor_data['species_by_genus'][selected_genus])
        else:
            species_options = ["All"]
        selected_species = st.selectbox("Flavor Species:", species_options)

    if selected_family != "All" or selected_genus != "All" or selected_species != "All":
        display_flavor_analysis(selected_family, selected_genus, selected_species)


def _target_flavor(family: str, genus: str, species: str):
    if species != "All":
        return species, "species"
    if genus != "All":
        return genus, "genus"
    if family != "All":
        return family, "family"
    return None, None


def get_cooccurring_flavors_for_flavor(target_flavor: str, taxonomy_level: str) -> Optional[List[Dict]]:
    """Companions of a flavor as P(B | A) from the full pair table.
    Returns None when the cache predates the per-flavor table."""
    data = load_cooccurrence_data() or {}
    level_data = data.get(f'{taxonomy_level}_level', {})
    by_flavor = level_data.get('by_flavor')
    if by_flavor is None:
        return None
    return by_flavor.get(target_flavor, [])


def display_flavor_analysis(family: str, genus: str, species: str):
    """Display results for the selected flavor"""
    st.subheader("Analysis for Selected Flavor")

    desc = []
    if family != "All":
        desc.append(f"Family: {family}")
    if genus != "All":
        desc.append(f"Genus: {genus}")
    if species != "All":
        desc.append(f"Species: {species}")
    st.write(f"**Selected Flavor:** {' → '.join(desc)}")

    target, level = _target_flavor(family, genus, species)
    if target is None:
        st.info("Please select a specific flavor to see analysis results.")
        return
    st.caption(f"Analysis is at the **{level}** level for **{target}** (levels are tested separately).")

    unit_type = st.radio("Show by:", [UNIT_TYPE_SINGULAR_LABELS[t] for t in UNIT_TYPES],
                         horizontal=True, key="flavor_unit_type").lower()
    rows = load_flavor_unit_rows(unit_type, level, target)

    st.subheader(f"🌍 Where {target} stands out")
    distinctive = sorted([r for r in rows if r.get('distinctive')],
                         key=lambda r: r.get('log_odds', 0), reverse=True)
    if distinctive:
        top = distinctive[:MAX_SENTENCES]
        plot_df = pd.DataFrame([
            {UNIT_TYPE_SINGULAR_LABELS[unit_type]: unit_label(unit_type, r['unit']),
             'Share of coffees': v, 'Group': g}
            for r in top
            for g, v in (('This group', r['share_in']), ('Other coffees', r['share_rest']))
        ])
        fig = px.bar(plot_df, x=UNIT_TYPE_SINGULAR_LABELS[unit_type], y='Share of coffees',
                     color='Group', barmode='group',
                     title=f"Share of coffees listing {target}")
        fig.update_layout(xaxis_tickangle=45, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        for r in top:
            line = f"- {sentence_for(r, unit_type)}"
            sup = support_text(r)
            if sup:
                line += f" — {sup}"
            st.markdown(line)
        if unit_type != 'seller':
            st.caption(SELLER_SUPPORT_NOTE)
        with st.expander("Evidence"):
            st.dataframe(evidence_frame(distinctive, unit_type, by_unit=True), hide_index=True)
    elif rows:
        st.info(f"No {unit_type} stands out for {target} at the {level} level.")
    else:
        st.info(f"No {unit_type} had enough coffees listing {target} to test "
                f"(needs a group of at least 10 coffees with at least 3 listing it).")

    if rows:
        with st.expander(f"All {UNIT_TYPE_PLURAL_LABELS[unit_type].lower()} tested for {target}"):
            ordered = sorted(rows, key=lambda r: r.get('log_odds', 0), reverse=True)
            tbl = evidence_frame(ordered, unit_type, by_unit=True)
            tbl['Stands out?'] = ["Yes" if r.get('distinctive') else "No" for r in ordered]
            st.dataframe(tbl, hide_index=True)

    # Co-occurrence
    st.subheader("🤝 Often listed together")
    companions = get_cooccurring_flavors_for_flavor(target, level)
    if companions is None:
        st.info(REGENERATE_MSG)
    elif companions:
        st.write(f"Flavors listed in coffees that also list **{target}**, compared with how often "
                 "they appear in all coffees with flavor notes:")
        for c in companions[:MAX_SENTENCES]:
            st.write(f"• {c['flavor']}: listed in {fmt_pct(c['p_b_given_a'])} of {target} coffees "
                     f"vs {fmt_pct(c['p_b'])} of coffees with flavor notes ({c['cooccurrence_count']} coffees)")
    else:
        st.info(f"Not enough coffees list {target} together with another flavor to report.")

    # Summary
    insights = []
    if distinctive:
        r0 = distinctive[0]
        insights.append(f"**{unit_label(unit_type, r0['unit'])}** is where {target} stands out most "
                        f"({fmt_pct(r0['share_in'])} of its coffees list it vs {fmt_pct(r0['share_rest'])} of others).")
    if companions:
        insights.append(f"**{target}** is most often listed together with **{companions[0]['flavor']}**.")
    if insights:
        st.subheader("💡 Key Insights")
        for i in insights:
            st.write(f"• {i}")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def render_compare_section():
    """Side-by-side comparison of 2-4 groups"""
    st.markdown("""
    **What you'll see here:** Side-by-side comparison of countries, regions or sellers.

    **How to interpret:**
    - **Coffees**: group size. More coffees means more reliable results.
    - **Distinctive flavors**: what stands out in each group at the chosen level, and which flavors are shared or unique.
    - **How different the flavor mixes are**: 0 means identical overall flavor mixes, 1 means completely different (family level).

    Use this to answer questions like "How does Colombian coffee differ from Ethiopian?"
    """)

    comparison_type = st.radio("Compare:", [UNIT_TYPE_PLURAL_LABELS[t] for t in UNIT_TYPES], key="compare_type")
    entity_type = {v: k for k, v in UNIT_TYPE_PLURAL_LABELS.items()}[comparison_type]
    level = level_selector("compare_level")
    available_entities = get_available_units(entity_type)

    if available_entities:
        selected_entities = st.multiselect(
            f"Select {comparison_type.lower()} to compare (max 4):",
            available_entities, max_selections=4,
            default=top_units_by_size(entity_type, available_entities, 2),
            format_func=lambda u: unit_label(entity_type, u), key=f"compare_units_{entity_type}")
        if len(selected_entities) >= 2:
            display_comparison(selected_entities, entity_type, level)
        else:
            st.info("Please select at least 2 to compare.")
    else:
        st.info(REGENERATE_MSG)


def display_comparison(entities: List[str], entity_type: str, level: str = 'genus'):
    """Display comparison between selected units"""
    st.subheader("Comparison Results")

    profiles = []
    for entity in entities:
        profile = load_unit_profile(entity, entity_type)
        if profile and profile.get('levels', {}).get(level):
            profiles.append(profile)
    if not profiles:
        st.warning("No profile data available for the selected groups")
        return

    cols = st.columns(len(profiles))
    for col, p in zip(cols, profiles):
        L = p['levels'][level]
        with col:
            st.write(f"**{p.get('label', p['unit_name'])}**")
            st.metric("Coffees", fmt_n(p.get('n_coffees', 'N/A')))
            st.metric(f"Distinctive {level} flavors", L.get('n_distinctive', 0))
            if L.get('flavors'):
                for row in L['flavors'][:3]:
                    st.write(f"- {sentence_for(row, entity_type, p['unit_name'])}")
            else:
                st.write(f"{NO_STANDOUT} at this level.")

    # Shared / unique distinctive flavors
    flavor_sets = {p.get('label', p['unit_name']): {f['flavor'] for f in p['levels'][level].get('flavors', [])}
                   for p in profiles}
    all_flavors = sorted(set().union(*flavor_sets.values())) if flavor_sets else []
    st.subheader("Shared and unique distinctive flavors")
    if all_flavors:
        table = []
        for fl in all_flavors:
            have = [name for name, s in flavor_sets.items() if fl in s]
            table.append({'Flavor': fl,
                          'Distinctive for': ", ".join(have),
                          'Shared?': "Shared" if len(have) > 1 else "Only one"})
        st.dataframe(pd.DataFrame(table).sort_values(['Shared?', 'Flavor']), hide_index=True)
    else:
        st.info(f"None of the selected groups has a distinctive flavor at the {level} level.")

    # Pairwise difference of overall flavor mix (family level)
    with_shares = [p for p in profiles if p.get('family_shares')]
    st.subheader("How different the flavor mixes are")
    if len(with_shares) >= 2:
        labels = [p.get('label', p['unit_name']) for p in with_shares]
        matrix = [[round(jsd_between(a['family_shares'], b['family_shares']), 3) for b in with_shares]
                  for a in with_shares]
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels))
        st.caption("0 = identical overall flavor mix, 1 = completely different. Measured on flavor families, "
                   "with each coffee counted equally and small groups pulled toward the market average.")
    else:
        st.info("Flavor-mix comparison not available.")
