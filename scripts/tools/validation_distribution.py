from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import unicodedata
import pandas as pd
import re


data = pd.read_excel("...")
diagnosis_dictionary = pd.read_excel("inputs/LN_Dx_dictionary_codes_20260824.xlsx")

diagnosis_dictionary_code = diagnosis_dictionary["Code"].tolist()
diagnosis_dictionary_diag = diagnosis_dictionary["Diagnosis"].tolist()

# --------------------------
# Disease Dictionary Creation
# --------------------------
disease_dict = {}

for x in zip(diagnosis_dictionary_code, diagnosis_dictionary_diag):
    disease_dict[x[1]] = x[0]


# ---------------------------
# Text Normalization
# ---------------------------
def normalize(text: str):
    text = unicodedata.normalize("NFKC", text)

    norm_chars = []

    prev_space = False

    for i, ch in enumerate(text):
        ch = ch.lower()

        if re.match(r"[a-z0-9]", ch):
            norm_chars.append(ch)
            prev_space = False

        else:
            if not prev_space:
                norm_chars.append(" ")
                prev_space = True

    normalized = "".join(norm_chars).strip()

    return normalized


# TODO:
synonyms = data["GT Report Diagnosis"]
valid = data["Container Duplicate"]
mask = synonyms.notna()

synonyms = synonyms[mask]
valid = valid[mask]

valid_syns = synonyms[valid]


print(f"Number of cases: {len(valid_syns)}")
print(f"Number of names: {len(np.unique(valid_syns))}")


normed_dict = {normalize(k): v for k, v in disease_dict.items()}
valid_syns = [normalize(s) for s in valid_syns]
valid_syns = [normalize(s) for s in valid_syns]
valid_syns = [s for s in valid_syns if s != "large b cell lymphoma"]

valid_codes = [str(normed_dict[d]) for d in valid_syns]

valid_codes = [str(normed_dict[d]) for d in valid_syns]

print(f"Number of codes: {len(np.unique(valid_codes))}")


# --- Count frequencies and normalize keys ---
counts = Counter(valid_syns)
counts = {str(k): v for k, v in counts.items()}

# --- Map each name to a group/code ---
name_to_group = {str(name): normed_dict[str(name)] for name in valid_syns}

# --- Sort unique groups numerically ---
unique_groups = sorted(set(name_to_group.values()))

# --- Assign consistent colors to groups ---
colors = np.vstack([plt.cm.tab20.colors, plt.cm.tab20b.colors, plt.cm.tab20c.colors])
group_to_index = {g: i for i, g in enumerate(unique_groups)}
# cmap = plt.cm.get_cmap('prism', len(unique_groups))
cmap = plt.cm.colors.ListedColormap(colors[: len(unique_groups)])
group_to_color = {g: cmap(i) for g, i in group_to_index.items()}

# --- Sort labels by group (code) then frequency ---
sorted_items = sorted(counts.items(), key=lambda x: (name_to_group[x[0]], x[1]))
labels, values = zip(*sorted_items)

# --- Assign colors per bar ---
colors = [group_to_color[name_to_group[label]] for label in labels]

# --- Plot ---
plt.figure(figsize=(17, 14))
bars = plt.barh(labels, values, color=colors)

# Optional: add value labels to bars
for bar in bars:
    plt.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        str(int(bar.get_width())),
        va="center",
        ha="left",
        fontsize=8,
    )

plt.xlabel("Count")
plt.ylabel("Name")
plt.title("Frequency of Diagnostic Names by Code")
plt.tight_layout()

ax = plt.gca()
ax.set_ylabel("Name", rotation=0)
ax.yaxis.set_label_coords(-0.04, 0.98)

# --- Invert y-axis so lowest code is at the top ---
ax.invert_yaxis()

# Add summary box
ax.text(
    0.87,
    0.95,
    f"#cases: {len(valid_syns)}\n#names: {len(np.unique(valid_syns))}\n#categs: {len(np.unique(valid_codes))}",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# --- Legend: show original numeric group codes ---
handles = [mpatches.Patch(color=group_to_color[g], label=f"{g}") for g in unique_groups]
plt.legend(handles=handles, title="Code", bbox_to_anchor=(0.87, 0.93), loc="upper left")

plt.show()

# Save the figure
plt.savefig("diagnostic_name_distribution.png", dpi=600, bbox_inches="tight")
